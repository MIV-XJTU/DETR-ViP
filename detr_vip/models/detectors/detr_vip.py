# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import random
import warnings
from typing import Dict, Optional, Tuple, Union, List
import types
import json

import torch
import torch.nn as nn

from torch import Tensor
import torch.distributed as dist

from mmengine.runner.amp import autocast
from mmengine.dataset import pseudo_collate
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList, DetDataSample
from mmdet.utils import ConfigType

from mmdet.models.layers import SinePositionalEncoding, DeformableDetrTransformerEncoder, DinoTransformerDecoder
from mmdet.models.layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from mmdet.models.detectors.dino import DINO

from ..prompt_encoder import VisualPromptEncoder
from ..utils.misc import gather_logits, nested_2Dtensor_from_tensor_list

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]


# def clean_label_name(name: str) -> str:
#     name = re.sub(r'\(.*\)', '', name)
#     name = re.sub(r'_', ' ', name)
#     name = re.sub(r'  ', ' ', name)
#     return name


# def chunks(lst: list, n: int) -> list:
#     """Yield successive n-sized chunks from lst."""
#     all_ = []
#     for i in range(0, len(lst), n):
#         data_index = lst[i:i + n]
#         all_.append(data_index)
#     counter = 0
#     for i in all_:
#         counter += len(i)
#     assert (counter == len(lst))

#     return all_


@MODELS.register_module()
class DETRViP(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 *args,
                 language_dim,
                 visual_prompt_encoder,
                 multiple_layer_align,
                 training_mode,
                 use_autocast=False,
                 **kwargs) -> None:
        self.language_dim = language_dim
        self.visual_prompt_encoder = visual_prompt_encoder
        self.training_mode = training_mode
        self.multiple_layer_align = multiple_layer_align
        kwargs['bbox_head']['num_fusion'] = kwargs['encoder']['num_fusion']
        kwargs['bbox_head']['num_layers'] = kwargs['encoder']['num_layers']
        if 'num_expansion' in kwargs['encoder']:
            kwargs['bbox_head']['num_expansion'] = kwargs['encoder']['num_expansion'] 
        super().__init__(*args, **kwargs)
        self.use_autocast = use_autocast

    def _init_layers(self) -> None:
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = MODELS.build(self.encoder)
        self.decoder = MODELS.build(self.decoder)

        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.text_feat_map = nn.Sequential(nn.Linear(
            self.language_dim,
            self.embed_dims,
            bias=True),
            nn.LayerNorm(self.embed_dims))
        self.visual_prompt_encoder = MODELS.build(self.visual_prompt_encoder)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map[0].bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map[0].weight.data)

        self.visual_prompt_encoder.init_weights()

    def forward_prompt_encoder(
        self,
        feat: Tensor,
        feat_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: OptSampleList = None,
        visual_prompts_dict: Dict = None,
        nested=True,
        **kwargs
    ):  
        if visual_prompts_dict is not None:
            visual_prompts_dict = self.visual_prompt_encoder(feat, feat_mask, spatial_shapes, level_start_index, valid_ratios, **visual_prompts_dict)
            visual_prompt_list, visual_prompt_label_list = visual_prompts_dict['visual_prompt_list'], visual_prompts_dict['visual_prompt_label_list']
            if nested:
                visual_prompts, visual_token_masks = nested_2Dtensor_from_tensor_list(visual_prompt_list).decompose()
                visual_prompt_labels, _ = nested_2Dtensor_from_tensor_list(visual_prompt_label_list).decompose()
                visual_prompts_dict = {
                    'visual_prompts' : visual_prompts,
                    'visual_prompt_labels' : visual_prompt_labels,
                    'visual_token_masks': visual_token_masks
                }
            else:
                visual_prompts_dict = {
                    'visual_prompt_list':visual_prompt_list,
                    'visual_prompt_label_list':visual_prompt_label_list
                }
        return visual_prompts_dict

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_prompts_dict: Dict,
        visual_prompts_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        visual_prompts_dict = self.forward_prompt_encoder(\
            visual_prompts_dict=visual_prompts_dict, \
            **encoder_inputs_dict)

        prompt_dict, batch_data_samples, align_dict, prompt_mode = \
            self.pre_encoder(text_prompts_dict, visual_prompts_dict, batch_data_samples)

        encoder_outputs_dict, encoder_memory_prompts = \
            self.forward_encoder(prompt_dict=prompt_dict, \
                cls_branches=self.bbox_head.cls_branches[self.decoder.num_layers:self.decoder.num_layers+self.encoder.num_layers],\
                prompt_mode=prompt_mode,
                **encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples, prompt_mode=prompt_mode)
        decoder_inputs_dict.update(tmp_dec_in)


        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict, 
        cls_branches=self.bbox_head.cls_branches[:self.decoder.num_layers])

        head_inputs_dict.update(decoder_outputs_dict)

        head_inputs_dict['prompt_mode'] = prompt_mode
        head_inputs_dict['align_info'] = align_dict
        return head_inputs_dict, batch_data_samples

    def pre_encoder(self, text_prompts_dict, visual_prompts_dict, batch_data_samples):
        if self.training:
            text_prompts, text_token_masks, text_prompt_labels = \
                text_prompts_dict['text_prompts'], text_prompts_dict['text_token_masks'], text_prompts_dict['text_prompt_labels']

            visual_prompts, visual_token_masks, visual_prompt_labels = \
                visual_prompts_dict['visual_prompts'], visual_prompts_dict['visual_token_masks'], visual_prompts_dict['visual_prompt_labels']
            
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    p = torch.rand(1).to(visual_token_masks.device)
                else:
                    p = torch.zeros(1).to(visual_token_masks.device)
                dist.broadcast(p, src=0)
                p = p.item()
            else:
                p = random.random()

            prompt_dict = {}

            if (p < 0.5 and self.training_mode == 'random') or self.training_mode == 'text':
                mode = 'text'
                prompts, labels, token_masks = text_prompts, text_prompt_labels, text_token_masks
            elif (p > 0.5 and self.training_mode == 'random') or self.training_mode == 'visual':
                mode = 'visual'
                prompts, labels, token_masks = visual_prompts, visual_prompt_labels, visual_token_masks

                prompts = gather_logits(prompts[~token_masks]) if dist.is_initialized() else prompts[~token_masks]
                labels = gather_logits(labels[~token_masks]) if dist.is_initialized() else labels[~token_masks]

                prompts_list = []
                for l in labels.unique():
                    prompts_l = prompts[labels == l].sum(0) / (labels == l).sum()
                    prompts_list.append(prompts_l)
                prompts = torch.stack(prompts_list, 0)
                labels = labels.unique()
                
                bs = len(batch_data_samples)
                prompts = prompts.unsqueeze(0).repeat(bs, 1, 1)
                labels = labels.unsqueeze(0).repeat(bs, 1)
                token_masks = torch.zeros_like(labels).bool()

            prompt_dict['embedded'] = prompts
            prompt_dict['token_masks'] = ~token_masks
            prompt_dict['position_ids'] = torch.zeros_like(token_masks).to(torch.int64)
            fuse_masks = []
            for data_sample, token_mask, label in zip(batch_data_samples, token_masks, labels):
                num_prompt = label.shape[0]
                gt_labels = data_sample.gt_instances.labels
                positive_maps = gt_labels.unsqueeze(1) == label.unsqueeze(0)
                data_sample.gt_instances.positive_maps = positive_maps.to(torch.float32)
                data_sample.gt_instances.token_mask = ~token_mask.unsqueeze(0).repeat(gt_labels.shape[0], 1)
                fuse_masks.append(positive_maps.any(0))
            fuse_masks = torch.stack(fuse_masks, 0).unsqueeze(1)
            prompt_dict['fuse_masks'] = fuse_masks

            align_dict = {}
            align_dict['visual_prompts'] = visual_prompts
            align_dict['text_prompts'] = text_prompts
            align_dict['visual_prompt_labels'] = visual_prompt_labels
            align_dict['text_prompt_labels'] = text_prompt_labels
            align_dict['visual_token_masks'] = ~visual_token_masks
            align_dict['text_token_masks'] = ~text_token_masks
            align_dict['mode'] = mode
            return prompt_dict, batch_data_samples, align_dict, mode

        if text_prompts_dict is not None:
            prompt_dict = {}
            text_prompts, text_token_masks, text_prompt_labels = \
                text_prompts_dict['text_prompts'], text_prompts_dict['text_token_masks'], text_prompts_dict['text_prompt_labels']
            prompts, labels, token_masks = text_prompts, text_prompt_labels, text_token_masks
            prompt_dict['embedded'] = prompts
            prompt_dict['token_masks'] = ~token_masks
            prompt_dict['position_ids'] = torch.zeros_like(token_masks).to(torch.int64)

            fuse_masks = []

            for data_sample, label in zip(batch_data_samples, labels):
                data_sample.token_positive_map = dict(zip((label + 1).tolist(), label.unsqueeze(-1).tolist()))
                gt_labels = data_sample.gt_instances.labels
                positive_maps = gt_labels.unsqueeze(1) == label.unsqueeze(0)
                fuse_masks.append(positive_maps.any(0))
            fuse_masks = torch.stack(fuse_masks, 0).unsqueeze(1)
            prompt_dict['fuse_masks'] = fuse_masks

            return prompt_dict, batch_data_samples, None, 'text'
        
        if visual_prompts_dict is not None:
            prompt_dict = {}
            visual_prompts, visual_token_masks, visual_prompt_labels = \
                visual_prompts_dict['visual_prompts'], visual_prompts_dict['visual_token_masks'], visual_prompts_dict['visual_prompt_labels']
            prompts, labels, token_masks = visual_prompts, visual_prompt_labels, visual_token_masks
            prompt_dict['embedded'] = prompts
            prompt_dict['token_masks'] = ~token_masks
            prompt_dict['position_ids'] = torch.zeros_like(token_masks).to(torch.int64)

            fuse_masks = []
            for data_sample, label in zip(batch_data_samples, labels):
                data_sample.token_positive_map = dict(zip((label + 1).tolist(), label.unsqueeze(-1).tolist()))
                gt_labels = data_sample.gt_instances.labels
                positive_maps = gt_labels.unsqueeze(1) == label.unsqueeze(0)
                fuse_masks.append(positive_maps.any(0))
            fuse_masks = torch.stack(fuse_masks, 0).unsqueeze(1)
            prompt_dict['fuse_masks'] = fuse_masks

            return prompt_dict, batch_data_samples, None, 'visual'

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        prompt_dict: Dict, cls_branches, prompt_mode:str) -> Dict:
        token_mask = prompt_dict['token_masks']
        memory, memory_prompt, enc_outputs_classes = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for prompt encoder
            memory_prompt=prompt_dict['embedded'],
            prompt_attention_mask=~token_mask,
            return_intermediate=True,
            cls_branches=cls_branches,
            prompt_mode=prompt_mode,
            fuse_masks=prompt_dict.pop('fuse_masks', None))
        encoder_outputs_dict = dict(
            memory=memory[-1],
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_prompt=memory_prompt[-1],
            token_mask=token_mask,
            enc_outputs_classes=enc_outputs_classes)
        return encoder_outputs_dict, memory_prompt

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_prompt: Tensor,
        token_mask: Tensor,
        enc_outputs_classes: Tensor,
        batch_data_samples: OptSampleList = None,
        prompt_mode: str = 'visual'
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers+self.encoder.num_layers](output_memory, memory_prompt,
                                     token_mask, prompt_mode)

        cls_out_features = enc_outputs_class.shape[-1]
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        intermediate_topk_score = [
            torch.gather(
                enc_outputs_cls, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features)) for enc_outputs_cls in enc_outputs_classes
        ]
        intermediate_topk_score = torch.stack(intermediate_topk_score, 0)
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_batch_data_samples = copy.deepcopy(batch_data_samples)
            if dn_batch_data_samples[0].dataset_mode == 'VG':
                for sample in dn_batch_data_samples:
                    sample.gt_instances.labels = torch.ones_like(sample.gt_instances.labels) * (self.bbox_head.num_classes - 1)
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(dn_batch_data_samples)
            del dn_batch_data_samples
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_prompt=memory_prompt,
            prompt_attention_mask=~token_mask,
            enc_out_score=topk_score,
            prompt_mode=prompt_mode
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            intermediate_enc_outputs_class=intermediate_topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_prompt'] = memory_prompt
        head_inputs_dict['token_mask'] = token_mask
        return decoder_inputs_dict, head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        device = batch_inputs.device

        text = [
            data_samples.text for data_samples in batch_data_samples
        ]

        text_prompt_labels = [
            torch.from_numpy(data_samples.text_prompt_labels) for data_samples in batch_data_samples
        ]

        text_embeddings = [torch.from_numpy(data_samples.text_prompts) for data_samples in batch_data_samples]

        text_embeddings, text_token_masks = nested_2Dtensor_from_tensor_list(text_embeddings).decompose()
        text_prompt_labels, _ = nested_2Dtensor_from_tensor_list(text_prompt_labels).decompose()

        text_prompt_labels = text_prompt_labels.to(device)
        text_embeddings = text_embeddings.to(device)
        text_token_masks = text_token_masks.to(device)
        
        if self.text_feat_map is not None:
            text_prompts = self.text_feat_map(text_embeddings)
        
        text_prompts_dict = {
            'text_prompts': text_prompts,
            'text_prompt_labels': text_prompt_labels,
            'text_token_masks': text_token_masks
        }

        visual_prompts_info = [
            {'visual_prompts':data_samples.visual_prompts, 
            'visual_prompt_labels':data_samples.visual_prompt_labels,
            'visual_prompt_types':data_samples.visual_prompt_types} for data_samples in batch_data_samples
        ]
        
        visual_prompts_dict = pseudo_collate(visual_prompts_info)
        visual_prompts_dict = {k:nested_2Dtensor_from_tensor_list(v).to(device) for k,v in visual_prompts_dict.items()}

        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)
        
        head_inputs_dict, batch_data_samples = self.forward_transformer(\
                                                    visual_features, \
                                                    text_prompts_dict,
                                                    visual_prompts_dict, \
                                                    batch_data_samples)

        head_inputs_dict['align_info']['dataset_mode'] = batch_data_samples[0].dataset_mode

        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-1])
        text_embedding_labels = text_prompt_labels.flatten()
        text_embedding_list = []
        for l in text_embedding_labels.unique():
            text_embedding_list.append(text_embeddings[text_embedding_labels == l].mean(0).unsqueeze(0))
        label_to_embeddings = dict(zip(text_embedding_labels.unique().tolist(), text_embedding_list))
        head_inputs_dict['align_info']['label_to_embeddings'] = label_to_embeddings
        
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def get_visual_prompts(self, batch, batch_data_samples):
        device = batch.device

        visual_prompts_info = [
            {'visual_prompts':data_samples.visual_prompts, 
            'visual_prompt_labels':data_samples.visual_prompt_labels,
            'visual_prompt_types':data_samples.visual_prompt_types} for data_samples in batch_data_samples
        ]

        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch)
        else:
            visual_features = self.extract_feat(batch)

        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            visual_features, batch_data_samples)

        visual_prompts_dict = pseudo_collate(visual_prompts_info)
        visual_prompts_dict = {k:nested_2Dtensor_from_tensor_list(v).to(device) for k,v in visual_prompts_dict.items()}

        visual_prompt_dict = self.forward_prompt_encoder(
                                visual_prompts_dict=visual_prompts_dict,
                                nested=False,
                                **encoder_inputs_dict)
        return visual_prompt_dict

    def predict(self, batch_inputs, batch_data_samples, prompt_dict, test_mode, rescale: bool = True):
        if test_mode == 'text':
            return self.predict_text_prompts(batch_inputs, batch_data_samples, prompt_dict, rescale)
        elif test_mode == 'visual-generic' or test_mode == 'visual-interactive':
            return self.predict_visual_generic(batch_inputs, batch_data_samples, prompt_dict, rescale)

    def predict_visual_generic(self, batch_inputs, batch_data_samples, visual_prompts_dict, rescale: bool = True):
        # image feature extraction
        bs = batch_inputs.shape[0]
        assert bs == 1
        visual_feats = self.extract_feat(batch_inputs)

        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            visual_feats, batch_data_samples)

        visual_prompt_list, visual_prompt_label_list = \
            visual_prompts_dict['visual_prompt_list'], visual_prompts_dict['visual_prompt_label_list']

        visual_prompts = torch.stack(visual_prompt_list, 0).unsqueeze(0).repeat(bs, 1, 1)
        visual_prompt_labels = torch.stack(visual_prompt_label_list, 0).unsqueeze(0).repeat(bs, 1)

        visual_prompt_label_map = dict(zip(list(range(len(visual_prompt_labels[0]))), visual_prompt_labels[0].tolist()))
        visual_prompt_labels = torch.arange(len(visual_prompt_labels[0]), device=visual_prompt_labels.device, dtype=visual_prompt_labels.dtype).unsqueeze(0)
        visual_token_masks = torch.zeros_like(visual_prompt_labels).bool()

        if visual_token_masks.shape[-1] == 1203:
            visual_token_masks = ~visual_token_masks
            with open('lvis_minival_valid_cls.json', 'r') as f:
                valid_cls_id = json.load(f)
            visual_token_masks[:, valid_cls_id] = False
            
        visual_prompts_dict = {
            'visual_prompts':visual_prompts,
            'visual_prompt_labels':visual_prompt_labels,
            'visual_token_masks':visual_token_masks
        }
        prompt_dict, batch_data_samples, align_dict, prompt_mode = self.pre_encoder(None, visual_prompts_dict, batch_data_samples)

        encoder_outputs_dict, _ = self.forward_encoder(
            prompt_dict=prompt_dict,
            cls_branches=self.bbox_head.cls_branches[self.decoder.num_layers:self.decoder.num_layers+self.encoder.num_layers],\
            prompt_mode=prompt_mode,
            **encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples, prompt_mode=prompt_mode)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict, 
                    cls_branches=self.bbox_head.cls_branches[:self.decoder.num_layers])
        head_inputs_dict.update(decoder_outputs_dict)

        head_inputs_dict['prompt_mode'] = prompt_mode

        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)

        for data_sample, pred_instances in zip(
                batch_data_samples, results_list):
            pred_labels = pred_instances.labels.tolist()
            pred_labels = [visual_prompt_label_map[_] for _ in pred_labels]
            pred_labels = torch.tensor(pred_labels, device=pred_instances.labels.device, dtype=pred_instances.labels.dtype)
            pred_instances.labels = pred_labels
            data_sample.pred_instances = pred_instances
        
        return batch_data_samples

    def predict_text_prompts(self, batch_inputs, batch_data_samples, text_prompts_dict, rescale: bool = True):
        device = batch_inputs.device
        bs = batch_inputs.shape[0]
        visual_feats = self.extract_feat(batch_inputs)

        text_prompts = [torch.from_numpy(data_samples.text_prompts) for data_samples in batch_data_samples]

        text_prompts, text_token_masks = nested_2Dtensor_from_tensor_list(text_prompts).decompose()
        text_prompt_labels = (~text_token_masks).cumsum(-1) - 1

        text_prompt_labels = text_prompt_labels.to(device)
        text_prompts = text_prompts.to(device)
        text_token_masks = text_token_masks.to(device)

        if text_token_masks.shape[-1] == 1203:
            text_token_masks = ~text_token_masks
            with open('lvis_minival_valid_cls.json', 'r') as f:
                valid_cls_id = json.load(f)
            text_token_masks[:, valid_cls_id] = False
            
        if self.text_feat_map is not None:
            text_prompts = self.text_feat_map(text_prompts)
        
        text_prompts_dict={
            'text_prompts':text_prompts,
            'text_prompt_labels':text_prompt_labels,
            'text_token_masks':text_token_masks
        }

        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            visual_feats, batch_data_samples)

        prompt_dict, batch_data_samples, align_dict, prompt_mode = self.pre_encoder(text_prompts_dict, None, batch_data_samples)

        encoder_outputs_dict, _ = self.forward_encoder(
            prompt_dict=prompt_dict,
            cls_branches=self.bbox_head.cls_branches[self.decoder.num_layers:self.decoder.num_layers+self.encoder.num_layers],\
            prompt_mode=prompt_mode,
            **encoder_inputs_dict)
        
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples, prompt_mode=prompt_mode)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict, 
                    cls_branches=self.bbox_head.cls_branches[:self.decoder.num_layers])

        head_inputs_dict.update(decoder_outputs_dict)

        head_inputs_dict['prompt_mode'] = prompt_mode

        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)

        for data_sample, pred_instances in zip(
                batch_data_samples, results_list):
            data_sample.pred_instances = pred_instances
        return batch_data_samples


    def get_visual_prompt_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='get_visual_prompts')  # type: ignore


    def val_step(self, data: Union[dict, tuple, list], prompt_dict, test_mode) -> list:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict', prompt_dict=prompt_dict, test_mode=test_mode)  # type: ignore

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        prompt_dict = data.pop("prompt_dict", None)
        test_mode = data.pop("test_mode", 'text')
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict', prompt_dict=prompt_dict, test_mode=test_mode)  # type: ignore

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        elif mode == 'get_visual_prompts':
            return self.get_visual_prompts(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
    
    def _run_forward(self, data: Union[dict, tuple, list],
                     mode: str, **kwargs) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self(**data, mode=mode, **kwargs)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode, **kwargs)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results