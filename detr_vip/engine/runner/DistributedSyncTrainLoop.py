# Copyright (c) MIV-XJTU. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List, Optional, Sequence, Tuple, Union
import copy
import itertools
import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from mmdet.registry import LOOPS

from mmengine.model import is_model_wrapper
from mmengine.runner import EpochBasedTrainLoop
from mmengine.runner.utils import calc_dynamic_intervals

from types import SimpleNamespace

@LOOPS.register_module()
class DistributedSyncTrainLoop(EpochBasedTrainLoop):
    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        self._runner = runner
        dataloader = self.build_dataloader(runner, dataloader)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        # if hasattr(self.dataloader.dataset, 'metainfo'):
        #     self.runner.visualizer.dataset_meta = \
        #         self.dataloader.dataset.metainfo
        # else:
        #     print_log(
        #         f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
        #         'metainfo. ``dataset_meta`` in visualizer will be '
        #         'None.',
        #         logger='current',
        #         level=logging.WARNING)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)
    
    def build_dataloader(self, runner, dataloader):
        if isinstance(dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            dataloader_cfg_vg = copy.deepcopy(dataloader)
            dataloader_cfg_od = copy.deepcopy(dataloader)

            dataloader_cfg_vg['dataset']['datasets'] = []
            dataloader_cfg_od['dataset']['datasets'] = []

            for ds in dataloader['dataset']['datasets']:
                if 'label_map_file' in ds and ds['label_map_file'] is not None:
                    dataloader_cfg_od['dataset']['datasets'].append(ds)
                else:
                    dataloader_cfg_vg['dataset']['datasets'].append(ds)
            dataloader = {}
            dataIter = {}
            if len(dataloader_cfg_vg['dataset']['datasets']) > 0:
                dataIter['VG'] = runner.build_dataloader(dataloader_cfg_vg, seed=runner.seed, diff_rank_seed=diff_rank_seed)
            if len(dataloader_cfg_od['dataset']['datasets']) > 0:
                dataIter['OD'] = runner.build_dataloader(dataloader_cfg_od, seed=runner.seed, diff_rank_seed=diff_rank_seed)
            dataloader['dataIter'] = dataIter
            dataloader['dataset'] = None
            dataloader['length'] = sum([len(v) for v in dataIter.values()])
            dataloader = SampleMultiDataloader(**dataloader)
            
        self.dataloader = dataloader
    
    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        
        total_step = len(self.dataloader)
        dataIter = self.dataloader.dataIter
        dataIterType = dict(zip(list(range(len(dataIter))), dataIter.keys()))
        dataIter = dict(zip(list(range(len(dataIter))), dataIter.values()))
        iterSampler = list(itertools.chain.from_iterable([[_] * len(dataIter[_]) for _ in dataIter]))
        
        random.shuffle(iterSampler)

        if self.runner.distributed:
            if dist.get_rank() == 0:
                dt_id = torch.tensor(iterSampler, device='cuda')
            else:
                dt_id = torch.empty(total_step, device='cuda').to(torch.int64)
            dist.broadcast(dt_id, src=0)
            iterSampler = dt_id.tolist()
        dataIter = {did:iter(dt) for did,dt in dataIter.items()}
        for idx in range(total_step):
            # if idx > 10:
            #     break
            data_batch = next(dataIter[iterSampler[idx]])
            datatype = dataIterType[iterSampler[idx]]
            for data_sample in data_batch['data_samples']:
                assert data_sample.dataset_mode == datatype
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

class SampleMultiDataloader(SimpleNamespace):
    def __len__(self):
        return self.length