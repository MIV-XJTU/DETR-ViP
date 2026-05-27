# Copyright (c) MIV-XJTU. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.

import random

import numpy as np

import torch
from torch import Tensor

from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, HorizontalBoxes

@TRANSFORMS.register_module()
class RandomSamplingVisualPrompt(BaseTransform):
    def __init__(self, select_mode='train', **kwargs):
        self.select_mode = select_mode
        super().__init__(**kwargs)

    def transform(self, results: dict) -> dict:
        boxes = results['gt_bboxes']            # numpy array
        if isinstance(boxes, Tensor):
            boxes = boxes.clone()
        elif isinstance(boxes, HorizontalBoxes):
            boxes = boxes.tensor.clone()
        labels = torch.from_numpy(results['gt_bboxes_labels']).clone()    # numpy array

        h, w = results['img_shape'] if self.select_mode == 'training' else results['ori_shape']
        factor = boxes.new_tensor([w, h, w, h])
        boxes = bbox_xyxy_to_cxcywh(boxes)
        boxes /= factor
        if boxes.numel() == 0:
            results['visual_prompts'] = boxes
            results['visual_prompt_labels'] = labels
            results['visual_prompt_types'] = torch.tensor([])
        else:
            labelset = labels.unique()
            prompts = []
            prompt_labels = []
            global_normalized_box = torch.tensor([0.5,0.5,1,1]).unsqueeze(0)
            # split = [0]
            prompt_type = []
            for _, l in enumerate(labelset):
                select_label = l
                select_inds = torch.where(labels == select_label)[0]
                spec_boxes = boxes[select_inds] # anno for spec class
                if self.select_mode == 'support':
                    num_prompt = len(spec_boxes)
                elif self.select_mode == 'training':
                    num_prompt = random.randint(1, len(spec_boxes)) if len(spec_boxes) > 0 else 0
                elif self.select_mode == 'test':
                    num_prompt = 1
                shuffle_boxid = list(range(len(spec_boxes)))
                random.shuffle(shuffle_boxid)
                prompt_id = shuffle_boxid[:num_prompt]
                visual_prompt = spec_boxes[prompt_id]
                visual_prompt = torch.cat((global_normalized_box.clone(), visual_prompt), 0)
                prompts.append(visual_prompt)
                prompt_labels.append(torch.ones(num_prompt + 1).to(torch.int64) * l)
                prompt_type.append(torch.tensor([0]+[1]*num_prompt))

            prompts = torch.cat(prompts, 0)
            prompt_labels = torch.cat(prompt_labels, 0)
            prompt_type = torch.cat(prompt_type, 0)

            results['visual_prompts'] = prompts
            results['visual_prompt_labels'] = prompt_labels
            results['visual_prompt_types'] = prompt_type
        # img = torch.from_numpy(results['img']).permute(2,0,1).unsqueeze(0) / 255.
        # img_name = results['img_path'].split('/')[-1]
        # draw_boxes_on_tensor(img, boxes, img_name)
        return results