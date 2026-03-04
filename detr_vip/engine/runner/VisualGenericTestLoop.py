import copy
from itertools import accumulate

from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather

from mmdet.engine.hooks.visualization_hook import DetVisualizationHook

from mmengine.evaluator import Evaluator
from mmengine.runner.amp import autocast
from mmengine.runner.loops import TestLoop, ValLoop, _update_losses
from mmengine.registry import LOOPS
from mmengine.model.wrappers import is_model_wrapper

@LOOPS.register_module()
class VisualGenericTestLoop(TestLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 test_mode: str = 'visual-generic',
                 fp16: bool = False):
        support_dataloader = dataloader.pop('support_dataloader')
        self.test_mode = test_mode
        # test_pipeline, support_pipeline = dataloader['dataset'].pop("pipeline")
        # testloader = copy.deepcopy(dataloader)
        # supportloader = copy.deepcopy(dataloader)

        # testloader['dataset']['pipeline'] = test_pipeline
        # supportloader['dataset']['pipeline'] = support_pipeline
        super().__init__(runner, dataloader, evaluator, fp16)
        diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
        self.support_dataloader = runner.build_dataloader(
                support_dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
    
    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        # clear test loss
        self.test_loss.clear()
        
        if self.test_mode == 'visual-generic':
            prompt_dict = self.prepare_visual_prompts()
        elif self.test_mode == 'text':
            prompt_dict = None
        elif self.test_mode == 'visual-interactive':
            prompt_dict = None
            # prompt_dict = self.prepare_text_prompts()
   
        for idx, data_batch in enumerate(self.dataloader):
            if self.test_mode == 'visual-interactive':
                if is_model_wrapper(self.runner.model):
                    prompt_dict = self.runner.model.module.get_visual_prompt_step(data_batch)
                else:
                    prompt_dict = self.runner.model.get_visual_prompt_step(data_batch)
                for k, v in prompt_dict.items():
                    prompt_dict[k] = [t for t in v[0]]
                if len(prompt_dict['visual_prompt_list']) == 0:
                    continue
            self.run_iter(idx, data_batch, prompt_dict)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        if self.test_loss:
            loss_dict = _parse_losses(self.test_loss, 'test')
            metrics.update(loss_dict)

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    @torch.no_grad()
    def prepare_visual_prompts(self, ):
        visual_prompt_list = []
        visual_prompt_label_list = []
        for idx, data_batch in enumerate(self.support_dataloader):
            if is_model_wrapper(self.runner.model):
                visual_prompts_dict = self.runner.model.module.get_visual_prompt_step(data_batch)
            else:
                visual_prompts_dict = self.runner.model.get_visual_prompt_step(data_batch)
            visual_prompt_list += visual_prompts_dict['visual_prompt_list']
            visual_prompt_label_list += visual_prompts_dict['visual_prompt_label_list']
        visual_prompts = torch.cat(visual_prompt_list, 0)
        visual_prompt_labels = torch.cat(visual_prompt_label_list, 0)
        
        if dist.is_initialized():
            visual_prompts = gather_prompts(visual_prompts)
            visual_prompt_labels = gather_prompts(visual_prompt_labels)
            
        visual_prompt_list = []
        visual_prompt_label_list = []
        for l in visual_prompt_labels.unique():
            label_prompt_inds = visual_prompt_labels == l
            label_prompts = visual_prompts[label_prompt_inds]
            visual_prompt_list.append(label_prompts.mean(0))
            visual_prompt_label_list.append(l)
        # visual_prompts = torch.stack(visual_prompt_list, 0)
        # visual_prompt_labels = torch.stack(visual_prompt_label_list, 0)
        return {"visual_prompt_list":visual_prompt_list, "visual_prompt_label_list":visual_prompt_label_list}

    # @torch.no_grad()
    # def prepare_text_prompts(self, ):
    #     text_prompts = list(self.dataloader.dataset.metainfo['classes'])
    #     text_prompts = self.runner.model.language_model(text_prompts, torch.cuda.current_device())
    #     text_prompt_labels = torch.arange(len(text_prompts)).to(text_prompts.device)
    #     return {"text_prompts":text_prompts, "text_prompt_labels":text_prompt_labels}

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], prompt_dict) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        data_batch['prompt_dict'] = prompt_dict
        data_batch['test_mode'] = self.test_mode

        with autocast(enabled=self.fp16):
            if is_model_wrapper(self.runner.model):
                outputs = self.runner.model.module.test_step(data_batch)
            else:
                outputs = self.runner.model.test_step(data_batch)

        outputs, self.test_loss = _update_losses(outputs, self.test_loss)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)

@LOOPS.register_module()
class MultiDatasetVisualGenericTestLoop(TestLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 test_mode: str = 'visual-generic',
                 fp16: bool = False):
        support_dataloader = dataloader.pop('support_dataloader')
        self.test_mode = test_mode

        self._runner = runner
        # Determine whether or not different ranks use different seed.
        diff_rank_seed = runner._randomness_cfg.get(
            'diff_rank_seed', False)
        datasets = dataloader['dataset']
        metainfo_list = []
        dataloader_list = []
        for ds in datasets:
            ds_dataloader = copy.deepcopy(dataloader)
            ds_dataloader['dataset'] = ds
            dl = runner.build_dataloader(
                ds_dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
            dataloader_list.append(dl)
            metainfo_list.append(ds['metainfo'])
        cumulative_sizes = list(accumulate([len(dl) for dl in dataloader_list]))
        for info in metainfo_list:
            info['cumulative_sizes'] = cumulative_sizes
        self.dataloader = DataLoaderList(dataloader_list)
        
        self.evaluator = runner.build_evaluator(evaluator)  # type: ignore

        self.evaluator.dataset_meta = metainfo_list
        self.runner.visualizer.dataset_meta = metainfo_list
        self.metainfo_list = metainfo_list
        self.fp16 = fp16
        self.test_loss: Dict[str, HistoryBuffer] = dict()

        diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
        support_dataloader_list = []
        support_dataset = support_dataloader['dataset']
        for ds in support_dataset:
            ds_dataloader = copy.deepcopy(dataloader)
            ds_dataloader['dataset'] = ds
            dl = runner.build_dataloader(
                ds_dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
            support_dataloader_list.append(dl)
        self.support_dataloader = DataLoaderList(support_dataloader_list)
    
    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        # clear test loss
        self.test_loss.clear()
        dataset_idx = 0
        idx = 0
        for support_dataloader, dataloader in zip(self.support_dataloader, self.dataloader):
            # for hook in self.runner.hooks:
            #     if isinstance(hook, DetVisualizationHook):
            self.runner.visualizer.dataset_meta = self.metainfo_list[dataset_idx]
                    # hook._visualizer.dataset_meta = dataloader.dataset.METAINFO
                    # ...
            if self.test_mode == 'visual-generic':
                prompt_dict = self.prepare_visual_prompts(support_dataloader)
            elif self.test_mode == 'text':
                prompt_dict = None
            elif self.test_mode == 'visual-interactive':
                prompt_dict = None
                # prompt_dict = self.prepare_text_prompts()
            # if idx > 10:
            #     break
            for _, data_batch in enumerate(dataloader):
                if self.test_mode == 'visual-interactive':
                    if is_model_wrapper(self.runner.model):
                        prompt_dict = self.runner.model.module.get_visual_prompt_step(data_batch)
                    else:
                        prompt_dict = self.runner.model.get_visual_prompt_step(data_batch)
                    for k, v in prompt_dict.items():
                        prompt_dict[k] = [t for t in v[0]]
                    if len(prompt_dict['visual_prompt_list']) == 0:
                        continue
                
                self.run_iter(idx, data_batch, prompt_dict)
                idx += 1
            dataset_idx += 1

        # compute metrics
        length_ds = sum([len(dl.dataset) for dl in self.dataloader])
        metrics = self.evaluator.evaluate(length_ds)

        if self.test_loss:
            loss_dict = _parse_losses(self.test_loss, 'test')
            metrics.update(loss_dict)

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    @torch.no_grad()
    def prepare_visual_prompts(self, support_dataloader):
        visual_prompt_list = []
        visual_prompt_label_list = []
        for idx, data_batch in enumerate(support_dataloader):
            if is_model_wrapper(self.runner.model):
                visual_prompts_dict = self.runner.model.module.get_visual_prompt_step(data_batch)
            else:
                visual_prompts_dict = self.runner.model.get_visual_prompt_step(data_batch)
            visual_prompt_list += visual_prompts_dict['visual_prompt_list']
            visual_prompt_label_list += visual_prompts_dict['visual_prompt_label_list']
        visual_prompts = torch.cat(visual_prompt_list, 0)
        visual_prompt_labels = torch.cat(visual_prompt_label_list, 0)

        visual_prompt_list = []
        visual_prompt_label_list = []
        for l in visual_prompt_labels.unique():
            label_prompt_inds = visual_prompt_labels == l
            label_prompts = visual_prompts[label_prompt_inds]
            visual_prompt_list.append(label_prompts.mean(0))
            visual_prompt_label_list.append(l)
        # visual_prompts = torch.stack(visual_prompt_list, 0)
        # visual_prompt_labels = torch.stack(visual_prompt_label_list, 0)
        return {"visual_prompt_list":visual_prompt_list, "visual_prompt_label_list":visual_prompt_label_list}

    # @torch.no_grad()
    # def prepare_text_prompts(self, ):
    #     text_prompts = list(self.dataloader.dataset.metainfo['classes'])
    #     text_prompts = self.runner.model.language_model(text_prompts, torch.cuda.current_device())
    #     text_prompt_labels = torch.arange(len(text_prompts)).to(text_prompts.device)
    #     return {"text_prompts":text_prompts, "text_prompt_labels":text_prompt_labels}

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], prompt_dict) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        data_batch['prompt_dict'] = prompt_dict
        data_batch['test_mode'] = self.test_mode

        with autocast(enabled=self.fp16):
            if is_model_wrapper(self.runner.model):
                outputs = self.runner.model.module.test_step(data_batch)
            else:
                outputs = self.runner.model.test_step(data_batch)

        outputs, self.test_loss = _update_losses(outputs, self.test_loss)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)

@LOOPS.register_module()
class VisualGenericValLoop(ValLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 val_mode: str = 'visual-generic',
                 fp16: bool = False):
        support_dataloader = dataloader.pop('support_dataloader')
        self.val_mode = val_mode
        # test_pipeline, support_pipeline = dataloader['dataset'].pop("pipeline")
        # testloader = copy.deepcopy(dataloader)
        # supportloader = copy.deepcopy(dataloader)

        # testloader['dataset']['pipeline'] = test_pipeline
        # supportloader['dataset']['pipeline'] = support_pipeline
        super().__init__(runner, dataloader, evaluator, fp16)
        diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
        self.support_dataloader = runner.build_dataloader(
                support_dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        # clear val loss
        self.val_loss.clear()

        if self.val_mode == 'visual-generic':
            prompt_dict = self.prepare_visual_prompts()
        elif self.val_mode == 'text':
            # prompt_dict = self.prepare_text_prompts()
            prompt_dict = None

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch, prompt_dict)
        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        if self.val_loss:
            loss_dict = _parse_losses(self.val_loss, 'val')
            metrics.update(loss_dict)

        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        # if self.val_mode == 'text':
        #     self.val_mode = 'visual-generic'
        # else:
        #     self.val_mode = 'text'

        return metrics

    @torch.no_grad()
    def prepare_visual_prompts(self, ):
        visual_prompt_list = []
        visual_prompt_label_list = []
        for idx, data_batch in enumerate(self.support_dataloader):
            if is_model_wrapper(self.runner.model):
                visual_prompts_dict = self.runner.model.module.get_visual_prompt_step(data_batch)
            else:
                visual_prompts_dict = self.runner.model.get_visual_prompt_step(data_batch)
            visual_prompt_list += visual_prompts_dict['visual_prompt_list']
            visual_prompt_label_list += visual_prompts_dict['visual_prompt_label_list']
        visual_prompts = torch.cat(visual_prompt_list, 0)
        visual_prompt_labels = torch.cat(visual_prompt_label_list, 0)
        
        if dist.is_initialized():
            visual_prompts = gather_prompts(visual_prompts)
            visual_prompt_labels = gather_prompts(visual_prompt_labels)

        visual_prompt_list = []
        visual_prompt_label_list = []
        for l in visual_prompt_labels.unique():
            label_prompt_inds = visual_prompt_labels == l
            label_prompts = visual_prompts[label_prompt_inds]
            visual_prompt_list.append(label_prompts.mean(0))
            visual_prompt_label_list.append(l)
        # visual_prompts = torch.stack(visual_prompt_list, 0)
        # visual_prompt_labels = torch.stack(visual_prompt_label_list, 0)
        return {"visual_prompt_list":visual_prompt_list, "visual_prompt_label_list":visual_prompt_label_list}

    # @torch.no_grad()
    # def prepare_text_prompts(self, ):
    #     text_prompts = list(self.dataloader.dataset.metainfo['classes'])
    #     if is_model_wrapper(self.runner.model):
    #         text_prompts = self.runner.model.module.language_model(text_prompts, torch.cuda.current_device())
    #     else:
    #         text_prompts = self.runner.model.language_model(text_prompts, torch.cuda.current_device())
    #     text_prompt_labels = torch.arange(len(text_prompts)).to(text_prompts.device)
    #     return {"text_prompts":text_prompts, "text_prompt_labels":text_prompt_labels}

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], prompt_dict):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            if is_model_wrapper(self.runner.model):
                outputs = self.runner.model.module.val_step(data_batch, prompt_dict, self.val_mode)
            else:
                outputs = self.runner.model.val_step(data_batch, prompt_dict, self.val_mode)

        outputs, self.val_loss = _update_losses(outputs, self.val_loss)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)

def gather_prompts(logits, detach=False):
    world_size = dist.get_world_size()
    local_n = torch.tensor([logits.shape[0]], dtype=torch.int64, device=logits.device)
    local_rank = dist.get_rank()

    # 收集所有进程的样本数
    all_n = torch.zeros(world_size, dtype=torch.int64, device=logits.device)
    dist.all_gather_into_tensor(all_n, local_n)

    # pad每个embeddings
    max_n = all_n.max().item()
    padded_logits = torch.zeros((max_n, *list(logits.shape[1:])), dtype=logits.dtype, device=logits.device)
    unpadded_flags = torch.zeros((max_n) , dtype=logits.dtype, device=logits.device).bool()

    padded_logits[:local_n] = logits.clone()
    unpadded_flags[:local_n] = True

    # 计算总样本数
    total_n = max_n * world_size

    # 预分配收集张量
    gathered_logits = torch.zeros((total_n, *list(logits.shape[1:])), dtype=logits.dtype, device=logits.device)
    gathered_flag = torch.zeros((total_n), dtype=logits.dtype, device=logits.device).bool()

    if detach:
        dist.all_gather_into_tensor(gathered_logits, padded_logits.detach())
        dist.all_gather_into_tensor(gathered_flag, unpadded_flags.detach())

        gathered_logits[local_rank * max_n:local_rank * max_n+local_n] = logits
    else:
        # dist.all_gather_into_tensor(gathered_logits, padded_logits)
        # dist.all_gather_into_tensor(gathered_flag, unpadded_flags)
        gathered_logits = torch.cat(all_gather(padded_logits))
        gathered_flag = torch.cat(all_gather(unpadded_flags))
        ...
    logits = gathered_logits[gathered_flag]
    return logits

class DataLoaderList:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders)

    def __getitem__(self, idx):
        return self.dataloaders[idx]

    def __iter__(self):
        return iter(self.dataloaders)

def gather_prompts(logits, detach=False):
    world_size = dist.get_world_size()
    local_n = torch.tensor([logits.shape[0]], dtype=torch.int64, device=logits.device)
    local_rank = dist.get_rank()

    # 收集所有进程的样本数
    all_n = torch.zeros(world_size, dtype=torch.int64, device=logits.device)
    dist.all_gather_into_tensor(all_n, local_n)

    # pad每个embeddings
    max_n = all_n.max().item()
    padded_logits = torch.zeros((max_n, *list(logits.shape[1:])), dtype=logits.dtype, device=logits.device)
    unpadded_flags = torch.zeros((max_n) , dtype=logits.dtype, device=logits.device).bool()

    padded_logits[:local_n] = logits.clone()
    unpadded_flags[:local_n] = True

    # 计算总样本数
    total_n = max_n * world_size

    # 预分配收集张量
    gathered_logits = torch.zeros((total_n, *list(logits.shape[1:])), dtype=logits.dtype, device=logits.device)
    gathered_flag = torch.zeros((total_n), dtype=logits.dtype, device=logits.device).bool()

    if detach:
        dist.all_gather_into_tensor(gathered_logits, padded_logits.detach())
        dist.all_gather_into_tensor(gathered_flag, unpadded_flags.detach())

        gathered_logits[local_rank * max_n:local_rank * max_n+local_n] = logits
    else:
        # dist.all_gather_into_tensor(gathered_logits, padded_logits)
        # dist.all_gather_into_tensor(gathered_flag, unpadded_flags)
        gathered_logits = torch.cat(all_gather(padded_logits))
        gathered_flag = torch.cat(all_gather(unpadded_flags))
        ...
    logits = gathered_logits[gathered_flag]
    return logits