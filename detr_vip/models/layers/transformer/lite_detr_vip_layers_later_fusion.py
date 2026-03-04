# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple
import math
import copy

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import ModuleList
from torch import Tensor

from mmdet.models.utils.vlfuse_helper import SingleScaleBiAttentionBlock
from mmdet.registry import MODELS
from mmdet.utils import ConfigType
from mmdet.models.layers.transformer.deformable_detr_layers import DeformableDetrTransformerEncoder, DetrTransformerEncoderLayer
from mmdet.models.layers.transformer.detr_layers import DetrTransformerEncoderLayer

from .detr_vip_layers import SelectiveBiAttentionBlock
try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None

class LiteDETRViPTransformerEncoderLayer(DetrTransformerEncoderLayer):

    def __init__(self, small_expand=False, encoder_scale=3, **kwargs):
        self.small_expand = small_expand
        self.encoder_scale = encoder_scale
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        if self.small_expand:
            self.ffn_ll = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, level_start_index: Tensor, 
                key: Tensor = None, **kwargs) -> Tensor:
        """Forward function of an encoder layer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor. has shape (bs, num_queries).
        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        if key is None:
            key = query
        query = self.self_attn(
            query=query,
            key=key,
            value=key,
            query_pos=query_pos,
            key_pos=query_pos,
            key_padding_mask=key_padding_mask,
            level_start_index=level_start_index,
            **kwargs)
        query = self.norms[0](query)
        # query = self.ffn(query)
        if self.small_expand:
            query_hl = query[:, level_start_index[4 - self.encoder_scale]:]
            query_ll = query[:, :level_start_index[4 - self.encoder_scale]]
            query_hl = self.ffn(query_hl)
            query_ll = self.ffn_ll(query_ll)
            query = torch.cat([query_ll, query_hl], 1)
        else:
            query = self.ffn(query)
        query = self.norms[1](query)

        return query

@MODELS.register_module()
class LiteDETRViPTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, prompt_layer_cfg: ConfigType, fusion_layer_cfg: ConfigType,
                 num_expansion=3, encoder_scale = 3, num_fusion = 3, **kwargs) -> None:
        self.prompt_layer_cfg = prompt_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.num_expansion = num_expansion
        self.encoder_scale = encoder_scale
        self.num_fusion = num_fusion
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        layers = []
        for i in range(self.num_expansion):
            for j in range(int(self.num_layers / self.num_expansion)-1):
                layers.append(LiteDETRViPTransformerEncoderLayer(**self.layer_cfg))
            layers.append(LiteDETRViPTransformerEncoderLayer(small_expand=True, encoder_scale=self.encoder_scale, **self.layer_cfg))
        self.layers = ModuleList(layers)

        if self.prompt_layer_cfg is not None:
            self.prompt_layers = ModuleList([
                DetrTransformerEncoderLayer(**self.prompt_layer_cfg)
                for _ in range(self.num_layers)
            ])
        else:
            self.prompt_layers = None

        if self.fusion_layer_cfg is not None:
            self.fusion_layers = ModuleList(
            [
                None
                for _ in range(self.num_layers - self.num_fusion)
            ] + 
            [
                SelectiveBiAttentionBlock(**self.fusion_layer_cfg)
                for _ in range(self.num_fusion)
            ])
        else:
            self.fusion_layers = None
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                if self.fusion_layers[i]:
                    self.fusion_layers[i] = checkpoint_wrapper(
                        self.fusion_layers[i])

        memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        memory_trans_norm = nn.LayerNorm(self.embed_dims)

        self.memory_trans_fc = nn.ModuleList([None for _ in range(self.num_layers - self.num_fusion)] + [copy.deepcopy(memory_trans_fc) for _ in range(self.num_fusion)])
        self.memory_trans_norm = nn.ModuleList([None for _ in range(self.num_layers - self.num_fusion)] + [copy.deepcopy(memory_trans_norm) for _ in range(self.num_fusion)])

    def gen_encoder_output_proposals(
            self, memory: Tensor, memory_mask: Tensor,
            spatial_shapes: Tensor, layer_id:int) -> Tuple[Tensor, Tensor]:
        """Generate proposals from encoded memory. The function will only be
        used when `as_two_stage` is `True`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).

        Returns:
            tuple: A tuple of transformed memory and proposals.

            - output_memory (Tensor): The transformed memory for obtaining
              top-k proposals, has shape (bs, num_feat_points, dim).
            - output_proposals (Tensor): The inverse-normalized proposal, has
              shape (batch_size, num_keys, 4) with the last dimension arranged
              as (cx, cy, w, h).
        """

        bs = memory.size(0)
        proposals = []
        _cur = 0  # start index in the sequence of the current level
        for lvl, HW in enumerate(spatial_shapes):
            H, W = HW

            if memory_mask is not None:
                mask_flatten_ = memory_mask[:, _cur:(_cur + H * W)].view(
                    bs, H, W, 1)
                valid_H = torch.sum(~mask_flatten_[:, :, 0, 0],
                                    1).unsqueeze(-1)
                valid_W = torch.sum(~mask_flatten_[:, 0, :, 0],
                                    1).unsqueeze(-1)
                scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)
            else:
                if not isinstance(HW, torch.Tensor):
                    HW = memory.new_tensor(HW)
                scale = HW.unsqueeze(0).flip(dims=[0, 1]).view(1, 1, 1, 2)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        # do not use `all` to make it exportable to onnx
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)).sum(
                -1, keepdim=True) == output_proposals.shape[-1]
        # inverse_sigmoid
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_mask is not None:
            output_proposals = output_proposals.masked_fill(
                memory_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        if memory_mask is not None:
            output_memory = output_memory.masked_fill(
                memory_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.memory_trans_fc[layer_id](output_memory)
        output_memory = self.memory_trans_norm[layer_id](output_memory)
        # [bs, sum(hw), 2]
        return output_memory, output_proposals

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_prompt: Tensor = None,
                prompt_attention_mask: Tensor = None,
                pos_prompt: Tensor = None,
                prompt_self_attention_masks: Tensor = None,
                position_ids: Tensor = None,
                return_intermediate = False,
                cls_branches=None,
                prompt_mode='',
                fuse_masks=None):
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            memory_prompt (Tensor, optional): Memory prompt. It has shape (bs,
                len_prompt, prompt_embed_dims).
            prompt_attention_mask (Tensor, optional): prompt token mask. It has
                shape (bs,len_prompt).
            pos_prompt (Tensor, optional): The positional encoding for prompt.
                Defaults to None.
            prompt_self_attention_masks (Tensor, optional): prompt self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): prompt position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        
        query_pos_hl = query_pos[:, level_start_index[4-self.encoder_scale]:]
        reference_points_hl = reference_points[:, level_start_index[4-self.encoder_scale]:]

        if self.prompt_layers:
            # generate pos_prompt
            bs, n_prompt, _ = memory_prompt.shape
            if pos_prompt is None and position_ids is None:
                pos_prompt = (
                    torch.arange(n_prompt,
                                 device=memory_prompt.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_prompt = get_prompt_sine_pos_embed(
                    pos_prompt, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_prompt = get_prompt_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)
        
        intermediate_memory = []
        intermediate_memory_prompt = []
        intermediate_enc_outputs_class = []
        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers[layer_id]:
                if cls_branches[layer_id]:
                    output_memory, output_proposals = self.gen_encoder_output_proposals(output, key_padding_mask, spatial_shapes, layer_id)
                    enc_outputs_class = cls_branches[layer_id](output_memory, memory_prompt, ~prompt_attention_mask, prompt_mode)
                    intermediate_enc_outputs_class.append(enc_outputs_class)
                    fuse_masks = enc_outputs_class.max(1)[0].unsqueeze(1) > math.log(0.1 / (1-0.1))
                else:
                    fuse_masks = torch.ones_like(fuse_masks).bool()
                # tgt = output[:, level_start_index[4-self.encoder_scale]:]
                # tgt, memory_prompt = self.fusion_layers[layer_id](
                #     visual_feature=tgt,
                #     lang_feature=memory_prompt,
                #     attention_mask_v=key_padding_mask[:, level_start_index[4-self.encoder_scale]:],
                #     attention_mask_l=prompt_attention_mask,
                #     adaptive_mask=fuse_masks,
                #     prompt_mode=prompt_mode
                #     # attention_mask_l=prompt_attention_mask,
                # )
                # output = torch.cat([output[:, :level_start_index[4-self.encoder_scale]], tgt], 1)

                output, memory_prompt = self.fusion_layers[layer_id](
                    visual_feature=output,
                    lang_feature=memory_prompt,
                    attention_mask_v=key_padding_mask,
                    attention_mask_l=prompt_attention_mask,
                    adaptive_mask=fuse_masks,
                    prompt_mode=prompt_mode
                    # attention_mask_l=prompt_attention_mask,
                )
            if (layer_id + 1) % (self.num_layers / self.num_expansion) == 0:
                output = layer(
                    query=output,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
            else:
                tgt = output[:, level_start_index[4-self.encoder_scale]:]
                src = output
                tgt = layer(
                    query=tgt,
                    key=src,
                    query_pos=query_pos_hl,
                    reference_points=reference_points_hl,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
                output = torch.cat([output[:, :level_start_index[4-self.encoder_scale]], tgt], 1)
            intermediate_memory.append(output)
            intermediate_memory_prompt.append(memory_prompt)
        if return_intermediate:
            return torch.stack(intermediate_memory), torch.stack(intermediate_memory_prompt), torch.stack(intermediate_enc_outputs_class)
        return output, memory_prompt, enc_outputs_class