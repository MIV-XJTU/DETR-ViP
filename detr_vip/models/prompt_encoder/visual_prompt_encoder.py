import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from typing import Tuple

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmdet.registry import MODELS
from mmdet.models.layers.transformer.dino_layers import DinoTransformerDecoder
from mmdet.models.layers.transformer.deformable_detr_layers import DeformableDetrTransformerDecoderLayer
from mmdet.models.layers.transformer.utils import MLP, coordinate_to_encoding, inverse_sigmoid
from mmdet.models.layers.transformer import MLP
from mmengine.model import BaseModule
from mmengine.model import ModuleList
from mmdet.utils import OptConfigType

@MODELS.register_module()
class VisualPromptEncoder(DinoTransformerDecoder):
    def __init__(self, prompt_embed_dim, layer_cfg, gather_prompts, **kwargs):
        # layer_cfg = kwargs.pop('transformerlayers')
        self.layer_cfg = layer_cfg
        super().__init__(layer_cfg=layer_cfg, **kwargs)
        self.global_embedding = nn.Embedding(1, prompt_embed_dim)
        self.context_embedding = nn.Embedding(1, prompt_embed_dim)

        self.nhead = self.layer_cfg['self_attn_cfg']['num_heads']

        # self.feat_map = nn.Sequential(nn.Linear(
        #     prompt_embed_dim,
        #     prompt_embed_dim,
        #     bias=True),
        #     nn.LayerNorm(prompt_embed_dim))
        self.gather_prompts = gather_prompts

    def init_weights(self) -> None:
        # nn.init.constant_(self.feat_map[0].bias.data, 0)
        # nn.init.xavier_uniform_(self.feat_map[0].weight.data)

        nn.init.xavier_uniform_(self.global_embedding.weight)
        nn.init.xavier_uniform_(self.context_embedding.weight)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

    def prepare_prompt(self, prompts, prompt_labels, prompt_type):
        box, mask = prompts.decompose()
        bs, num_prompt, _ = box.shape
        

        prompt_type = prompt_type.tensors

        global_batch, global_inds = torch.where(prompt_type == 0)
        context_batch, context_inds = torch.where(prompt_type == 1)
        
        prompt_embedding = self.global_embedding.weight.unsqueeze(1).repeat(bs, box.shape[1], 1)
        prompt_embedding[context_batch, context_inds] = self.context_embedding.weight

        label_id = prompt_labels.tensors[global_batch, global_inds]

        attn_mask = []

        # position_ids = torch.zeros(bs, num_prompt).to(prompt_embedding.device)
        for bi in range(bs):
            pt = prompt_type[bi]
            m = mask[bi]

            si = torch.cat((torch.where(pt == 0)[0], (~m).sum()[None]), 0)

            # region = tuple(zip(si.tolist()[:-1], si.tolist()[1:]))
            # for start, end in region:
            #     position_ids[bi, start:end] = torch.arange(0, end-start).to(prompt_embedding.device)

            attn_map = ~(torch.eye(num_prompt).bool())
            for i in range(len(si) - 1):
                attn_map[si[i]:si[i+1], si[i]:si[i+1]] = False

            attn_mask.append(attn_map.clone())
        
        if num_prompt > 0:
            attn_mask = torch.stack(attn_mask, 0)
            # attn_mask = attn_mask.unsqueeze(0).repeat(self.nhead, 1, 1, 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1)
            attn_mask = attn_mask.reshape(-1, num_prompt, num_prompt)
            attn_mask = attn_mask.bool().to(prompt_labels.device)
        else:
            attn_mask = None
        
        # self.prompt_decoder.layers[0].self_attn.num_heads
        return prompt_embedding, box, mask, attn_mask, global_batch, global_inds, label_id
        # return prompt_embedding, box, mask, attn_mask, global_batch, global_inds, label_id, position_ids

    def forward(self, feat, feat_mask, spatial_shapes, level_start_index, valid_ratios,
                visual_prompts, visual_prompt_labels, visual_prompt_types):

        prompt_embedding, reference_points, visual_padding_mask, self_attn_mask, global_batch, global_inds, label_id = self.prepare_prompt(visual_prompts, visual_prompt_labels, visual_prompt_types)
        bs, num_vp = prompt_embedding.shape[:2]
        hs_prompt, references_prompt = self.forward_encoder(query=prompt_embedding,
                        value=feat,
                        key_padding_mask=feat_mask, 
                        self_attn_mask=self_attn_mask,
                        reference_points=reference_points,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        valid_ratios=valid_ratios,
                        reg_branches=None)
        # hs_prompt = self.feat_map(hs_prompt)
        visual_prompts = hs_prompt[-1][global_batch, global_inds]
        # visual_prompts = self.feat_map(visual_prompts)

        visual_prompt_list = []
        visual_prompt_label_list = []
        for b in range(bs):
            batch_inds = global_batch == b
            batch_labels = label_id[batch_inds]
            batch_visual_prompts = visual_prompts[batch_inds]
            visual_prompt_list.append(batch_visual_prompts)
            visual_prompt_label_list.append(batch_labels)
        
        return {"visual_prompt_list":visual_prompt_list, "visual_prompt_label_list":visual_prompt_label_list}
        # return {"visual_prompts":hs_prompt, "visual_prompt_position_ids":position_ids, "visual_prompt_attn_mask":self_attn_mask.reshape(-1, self.nhead, num_vp, num_vp)[:, -1], "visual_padding_mask":visual_padding_mask}

    def forward_encoder(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points

class VisualPromptEncoderLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        self.cross_attn_cfg = cross_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """

        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query

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
        # dist.all_gather_into_tensor(gathered_logits, padded_logits)
        # dist.all_gather_into_tensor(gathered_flag, unpadded_flags)
    else:
        gathered_logits = torch.cat(all_gather(padded_logits))
        gathered_flag = torch.cat(all_gather(unpadded_flags))
        ...

    logits = gathered_logits[gathered_flag]
    return logits