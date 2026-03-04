# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import List, Optional
import pickle

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from .transforms import MapTextToEmbedding, RandomSamplingNegPosToList
from mmdet.datasets.base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class ODVGStableDataset(BaseDetDataset):
    """object detection and visual grounding dataset."""

    def __init__(self,
                 *args,
                 data_root: str = '',
                 label_map_file: Optional[str] = None,
                 need_text: bool = True,
                 text_cache_file: Optional[str] = None, 
                 neg_phrase_list: Optional[str] = None, 
                 **kwargs) -> None:
        self.dataset_mode = 'VG'
        self.need_text = need_text
        
        if label_map_file:
            label_map_file = osp.join(data_root, label_map_file)
            with open(label_map_file, 'r') as file:
                self.label_map = json.load(file)
            self.dataset_mode = 'OD'
        super().__init__(*args, data_root=data_root, **kwargs)

        with open(text_cache_file, 'rb') as f:
            self.text_cache = pickle.load(f)
        if self.dataset_mode == 'VG':
            text_list = sorted(list(self.text_cache.keys()))
            self.text_to_label = dict(zip(text_list, list(range(len(text_list)))))
            
            with open(neg_phrase_list, 'r') as f:
                self.neg_phrase_list = json.load(f)

            for transforms in self.pipeline.transforms:
                if isinstance(transforms, RandomSamplingNegPosToList):
                    transforms.set_text_to_label(self.text_to_label)
                    transforms.set_neg_phrase_list(self.neg_phrase_list)

        for transforms in self.pipeline.transforms:
            if isinstance(transforms, MapTextToEmbedding):
                transforms.set_text_cache(self.text_cache)
        assert self.return_classes is True

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None or not data['data_samples'].gt_instances.labels.numel() > 0:
                idx = self._rand_another()
                continue
            return data

    def load_data_list(self) -> List[dict]:
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            with open(local_path, 'r') as f:
                data_list = [json.loads(line) for line in f]

        out_data_list = []
        for data in data_list:
            data_info = {}
            img_path = osp.join(self.data_prefix['img'], data['filename'])
            data_info['img_path'] = img_path
            data_info['height'] = data['height']
            data_info['width'] = data['width']
            if self.dataset_mode == 'OD':
                if self.need_text:
                    data_info['text'] = self.label_map
                anno = data.get('detection', {})
                instances = [obj for obj in anno.get('instances', [])]
                bboxes = [obj['bbox'] for obj in instances]
                bbox_labels = [str(obj['label']) for obj in instances]

                instances = []
                for bbox, label in zip(bboxes, bbox_labels):
                    instance = {}
                    x1, y1, x2, y2 = bbox
                    inter_w = max(0, min(x2, data['width']) - max(x1, 0))
                    inter_h = max(0, min(y2, data['height']) - max(y1, 0))
                    if inter_w * inter_h == 0:
                        continue
                    if (x2 - x1) < 1 or (y2 - y1) < 1:
                        continue
                    instance['ignore_flag'] = 0
                    instance['bbox'] = bbox
                    instance['bbox_label'] = int(label)
                    instances.append(instance)
                data_info['instances'] = instances
                data_info['dataset_mode'] = self.dataset_mode
                # data_info['text_cache'] = self.text_cache
                out_data_list.append(data_info)
            else:
                anno = data['grounding']
                data_info['text'] = anno['caption']
                regions = anno['regions']

                instances = []
                phrases = {}
                for i, region in enumerate(regions):
                    bbox = region['bbox']
                    phrase = region['phrase']
                    tokens_positive = region['tokens_positive']
                    if not isinstance(bbox[0], list):
                        bbox = [bbox]
                    for box in bbox:
                        instance = {}
                        x1, y1, x2, y2 = box
                        inter_w = max(0, min(x2, data['width']) - max(x1, 0))
                        inter_h = max(0, min(y2, data['height']) - max(y1, 0))
                        if inter_w * inter_h == 0:
                            continue
                        if (x2 - x1) < 1 or (y2 - y1) < 1:
                            continue
                        instance['ignore_flag'] = 0
                        instance['bbox'] = box
                        instance['bbox_label'] = i
                        phrases[i] = {
                            'phrase': phrase,
                            'tokens_positive': tokens_positive
                        }
                        instances.append(instance)
                data_info['instances'] = instances
                data_info['phrases'] = phrases
                data_info['dataset_mode'] = self.dataset_mode
                out_data_list.append(data_info)

        del data_list
        return out_data_list
