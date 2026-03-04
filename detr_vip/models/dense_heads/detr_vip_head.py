import copy
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather

from mmcv.cnn import Linear
from mmdet.models.losses import QualityFocalLoss
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import InstanceList, reduce_mean, OptInstanceList
from mmengine.structures import InstanceData

from mmdet.models.layers import inverse_sigmoid
from mmdet.models.dense_heads.grounding_dino_head import GroundingDINOHead
from mmdet.models.dense_heads.dino_head import DINOHead
from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores
from mmdet.models.utils import multi_apply

from ..utils.misc import gather_logits

class AdaptiveContrastiveEmbed(nn.Module):
    """text visual ContrastiveEmbed layer.

    Args:
        max_text_len (int, optional): Maximum length of text.
        log_scale (Optional[Union[str, float]]):  The initial value of a
          learnable parameter to multiply with the similarity
          matrix to normalize the output.  Defaults to 0.0.
          - If set to 'auto', the similarity matrix will be normalized by
            a fixed value ``sqrt(d_c)`` where ``d_c`` is the channel number.
          - If set to 'none' or ``None``, there is no normalization applied.
          - If set to a float number, the similarity matrix will be multiplied
            by ``exp(log_scale)``, where ``log_scale`` is learnable.
        bias (bool, optional): Whether to add bias to the output.
          If set to ``True``, a learnable bias that is initialized as -4.6
          will be added to the output. Useful when training from scratch.
          Defaults to False.
    """

    def __init__(self,
                 log_scale: Optional[Union[str, float]] = None,
                 bias: bool = False,
                 memory_dim = 256):
        super().__init__()
        self.log_scale = log_scale
        memory_feat_map = nn.Sequential(
            nn.Linear(memory_dim, memory_dim), 
            nn.LayerNorm(memory_dim)
            )
        self.visual_memory_feat_map = copy.deepcopy(memory_feat_map)
        self.text_memory_feat_map = copy.deepcopy(memory_feat_map)
        if isinstance(log_scale, float):
            self.log_scale = nn.Parameter(
                torch.Tensor([float(log_scale)]), requires_grad=True)
        elif log_scale not in ['auto', 'none', None]:
            raise ValueError(f'log_scale should be one of '
                             f'"auto", "none", None, but got {log_scale}')

        self.text_bias = self.vis_bias = None
        if bias:
            bias_value = -math.log((1 - 0.01) / 0.01)
            self.text_bias = nn.Parameter(
                torch.Tensor([bias_value]), requires_grad=True)
            
            self.vis_bias = nn.Parameter(
                torch.Tensor([bias_value]), requires_grad=True)

    def forward(self, visual_feat: Tensor, prompt_feat: Tensor,
                prompt_token_mask: Tensor, prompt_type: str) -> Tensor:
        """Forward function.

        Args:
            visual_feat (Tensor): Visual features.
            text_feat (Tensor): Text features.
            text_token_mask (Tensor): A mask used for text feats.

        Returns:
            Tensor: Classification score.
        """
        if prompt_type == 'text':
            visual_feat = self.text_memory_feat_map(visual_feat) + 0. * self.visual_memory_feat_map(visual_feat)
        elif prompt_type == 'visual':
            visual_feat = 0. * self.text_memory_feat_map(visual_feat) + self.visual_memory_feat_map(visual_feat)
        res = visual_feat @ prompt_feat.transpose(-1, -2)
        if isinstance(self.log_scale, nn.Parameter):
            res = res * self.log_scale.exp()
        elif self.log_scale == 'auto':
            # NOTE: similar to the normalizer in self-attention
            res = res / math.sqrt(visual_feat.shape[-1])
        if prompt_type == 'text' and self.text_bias is not None:
            res = res + self.text_bias + self.vis_bias * 0
        if prompt_type == 'visual' and self.vis_bias is not None:
            res = res + self.vis_bias + self.text_bias * 0
        # if self.bias is not None:
        #     res = res + self.bias
        res.masked_fill_(~prompt_token_mask[:, None, :], float('-inf'))
        return res

@MODELS.register_module()
class DETRViPHead(DINOHead):
    def __init__(self, contrastive_cfg, loss_align=dict(type='InfoNCELoss',temperature=0.1), loss_distill=dict(type='InfoNCELoss',temperature=0.1), num_layers=6, num_fusion=3, **kwargs):
        self.contrastive_cfg = contrastive_cfg
        self.num_fusion = num_fusion
        self.num_layers = num_layers
        super().__init__(**kwargs)
        self.loss_align = MODELS.build(loss_align)
        self.loss_distill = MODELS.build(loss_distill)
        
    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        fc_cls = AdaptiveContrastiveEmbed(**self.contrastive_cfg)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        # NOTE: due to the fc_cls is a contrastive embedding and don't
        # have any trainable parameters,we do not need to copy it.
        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                # [fc_cls for _ in range(6)]+[None for _ in range(2)]+[fc_cls for _ in range(5)])
                [fc_cls for _ in range(6)]\
                +[None for _ in range(self.num_layers-self.num_fusion)]\
                +[fc_cls for _ in range(self.num_fusion+1)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList(
                # [copy.deepcopy(fc_cls) for _ in range(6)]+[None for _ in range(2)]+[copy.deepcopy(fc_cls) for _ in range(5)])
                [copy.deepcopy(fc_cls) for _ in range(6)]\
                +[None for _ in range(self.num_layers-self.num_fusion)]\
                +[copy.deepcopy(fc_cls) for _ in range(self.num_fusion+1)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        cls_out_channels = cls_score.shape[-1]
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
        gt_bboxes = gt_instances.bboxes

        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # Major changes. The labels are 0-1 binary labels for each bbox
        # and text tokens.
        labels = gt_bboxes.new_full((num_bboxes, cls_out_channels),
                                    0,
                                    dtype=torch.float32)
        labels[pos_inds] = gt_instances.positive_maps[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def forward(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        memory_prompt: Tensor,
        token_mask: Tensor,
        prompt_mode: str
    ) -> Tuple[Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (List[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            memory_prompt (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_token_mask (Tensor): Text token mask. It has shape (bs,
                len_text).

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state,
                                                        memory_prompt,
                                                        token_mask, prompt_mode)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords

    def losses_prompts(self, align_info):
        losses_prompts = {}

        visual_prompts = align_info['visual_prompts']
        text_prompts = align_info['text_prompts']

        visual_prompt_labels = align_info['visual_prompt_labels']
        text_prompt_labels = align_info['text_prompt_labels']
        
        visual_token_masks = align_info['visual_token_masks']
        text_token_masks = align_info['text_token_masks']

        dataset_mode = align_info['dataset_mode']
    
        visual_prompt_list = []
        text_prompt_list = []
        visual_label_list = []
        text_label_list = []
        similarity = []
        softlabel = []

        losses_align_items = []
        for vp, tp, vl, tl, vtm, ttm in \
            zip(visual_prompts, text_prompts, visual_prompt_labels, text_prompt_labels, visual_token_masks, text_token_masks):
            # if dataset_mode == 'VG':
            #     vp, tp, vl, tl = vp[vtm], tp[:vtm.shape[0]][vtm], vl[vtm], tl[:vtm.shape[0]][vtm]
            # else:
            vp, tp, vl, tl = vp[vtm], tp[ttm], vl[vtm], tl[ttm]
            loss_align = self.loss_align(vp, tp, vl.unsqueeze(-1) == tl.unsqueeze(0))
            losses_align_items.append(loss_align)

        losses_align_items = torch.cat(losses_align_items, 0)

        losses_prompts['loss_align'] = losses_align_items.mean()

        # text_embeddings = align_info['text_embeddings']
        
        valid_visual_prompts = visual_prompts[visual_token_masks]
        valid_visual_prompt_labels = visual_prompt_labels[visual_token_masks]

        label_to_embeddings = align_info['label_to_embeddings']
        valid_text_embeddings = []
        for l in valid_visual_prompt_labels.tolist():
            valid_text_embeddings.append(label_to_embeddings[l])
        valid_text_embeddings = torch.cat(valid_text_embeddings, 0)

        if dist.is_initialized():
            valid_visual_prompts = gather_logits(valid_visual_prompts, detach=True)
            valid_visual_prompt_labels = gather_logits(valid_visual_prompt_labels, detach=True)
            valid_text_embeddings = gather_logits(valid_text_embeddings, detach=True)

        loss_distill = self.loss_distill(valid_visual_prompts, valid_text_embeddings)

        losses_prompts["loss_distill"] = loss_distill
        return losses_prompts
        

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             memory_prompt: Tensor, token_mask: Tensor,
             enc_outputs_class: Tensor, intermediate_enc_outputs_class: Tensor, 
             enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int],
             align_info: Dict, prompt_mode:str) -> dict:
        losses_prompts = self.losses_prompts(align_info)

        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references, memory_prompt, token_mask, prompt_mode)
        self.token_masks = token_mask
        loss_inputs = outs + (enc_outputs_class, intermediate_enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)

        losses.update(losses_prompts)
        return losses

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        inter_enc_cls_score: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # extract denoising and matching part of outputs
        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds, dn_meta)

        loss_dict = self.loss_by_feat_underlying(
            all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        # NOTE DETRHead.loss_by_feat but not DeformableDETRHead.loss_by_feat
        # is called, because the encoder loss calculations are different
        # between DINO and DeformableDETR.

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            enc_loss_cls, inter_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, inter_enc_cls_score, enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou
            for i, enc_loss_cls_i in enumerate(inter_loss_cls):
                loss_dict[f'e{i}.enc_loss_cls'] = enc_loss_cls_i

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
                                  dn_losses_iou[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
        return loss_dict

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                memory_prompt: Tensor,
                token_mask: Tensor,
                batch_data_samples: SampleList,
                prompt_mode: str,
                rescale: bool = True,
                **kwargs) -> InstanceList:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (List[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            memory_prompt (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_token_mask (Tensor): Text token mask. It has shape (bs,
                len_text).
            batch_data_samples (SampleList): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            InstanceList: Detection results of each image
                after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_token_positive_maps = [
            data_samples.token_positive_map
            for data_samples in batch_data_samples
        ]

        outs = self(hidden_states, references, memory_prompt, token_mask, prompt_mode)

        predictions = self.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_positive_maps,
            rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        batch_img_metas: List[Dict],
                        batch_token_positive_maps: Optional[List[dict]] = None,
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            all_layers_cls_scores (Tensor):  Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (num_decoder_layers, bs, num_queries,
                4) with the last dimension arranged as (cx, cy, w, h).
            batch_img_metas (List[Dict]): _description_
            batch_token_positive_maps (list[dict], Optional): Batch token
                positive map. Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]
        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            token_positive_maps = batch_token_positive_maps[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   token_positive_maps,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list
    
    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                token_positive_maps: dict,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].
            token_positive_maps (dict): Token positive map.
            img_meta (dict): Image meta info.
            rescale (bool, optional): If True, return boxes in original image
                space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        img_shape = img_meta['img_shape']

        if token_positive_maps is not None:
            cls_score = convert_grounding_to_cls_scores(
                logits=cls_score.sigmoid()[None],
                positive_maps=[token_positive_maps])[0]
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            num_classes = cls_score.shape[-1]
            det_labels = indexes % num_classes
            bbox_index = indexes // num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            cls_score = cls_score.sigmoid()
            scores, _ = cls_score.max(-1)
            scores, indexes = scores.topk(max_per_img)
            bbox_pred = bbox_pred[indexes]
            det_labels = scores.new_zeros(scores.shape, dtype=torch.long)

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))
        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        return results

    def loss_by_feat_single(self, cls_scores: Tensor, inter_cls_score: Tensor, bbox_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)
        cls_out_channels = cls_scores.shape[-1]
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        with torch.no_grad():
            cls_reg_targets = self.get_targets(cls_scores_list,
                                               bbox_preds_list,
                                               batch_gt_instances,
                                               batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # ===== this change =====
        # Loss is not computed for the padded regions of the text.
        assert (self.token_masks.dim() == 2)
        text_masks = self.token_masks.new_zeros(
            (self.token_masks.size(0), cls_out_channels))
        text_masks[:, :self.token_masks.size(1)] = self.token_masks
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, cls_scores.size(1), 1)
        cls_scores = torch.masked_select(cls_scores, text_mask).contiguous()
        if inter_cls_score is not None:
            inter_cls_score = [torch.masked_select(inter_cls, text_mask).contiguous() for inter_cls in inter_cls_score]
        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[...,
                                      None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            raise NotImplementedError(
                'QualityFocalLoss for GroundingDINOHead is not supported yet.')
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        if inter_cls_score is not None:
            inter_loss_cls = [
                self.loss_cls(inter_cls, labels, label_weights, avg_factor=cls_avg_factor) for inter_cls in inter_cls_score
            ]
        else:
            inter_loss_cls = None
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        return loss_cls, inter_loss_cls, loss_bbox, loss_iou

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        cls_out_channels = dn_cls_scores.shape[-1]
        cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                              batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        # ===== this change =====
        # Loss is not computed for the padded regions of the text.
        assert (self.token_masks.dim() == 2)
        text_masks = self.token_masks.new_zeros(
            (self.token_masks.size(0), cls_out_channels))
        text_masks[:, :self.token_masks.size(1)] = self.token_masks
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, dn_cls_scores.size(1), 1)
        cls_scores = torch.masked_select(dn_cls_scores, text_mask).contiguous()
        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[...,
                                      None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)
        # =======================

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            if isinstance(self.loss_cls, QualityFocalLoss):
                raise NotImplementedError('QualityFocalLoss is not supported')
            else:
                loss_cls = self.loss_cls(
                    cls_scores,
                    labels,
                    label_weights,
                    avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def _get_dn_targets_single(self, gt_instances: InstanceData,
                               img_meta: dict, dn_meta: Dict[str,
                                                             int]) -> tuple:
        """Get targets in denoising part for one image.

        Args:
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        cls_out_channels = gt_instances.token_mask.shape[-1]
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = dn_meta['num_denoising_queries']
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(
                num_groups, dtype=torch.long, device=device)
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = \
                gt_bboxes.new_tensor([], dtype=torch.long)

        neg_inds = pos_inds + num_queries_each_group // 2
        # label targets
        # this change
        labels = gt_bboxes.new_full((num_denoising_queries, cls_out_channels),
                                    0,
                                    dtype=torch.float32)
        labels[pos_inds] = gt_instances.positive_maps[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        # bbox targets
        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)
                
    def loss_by_feat_underlying(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_layers_cls_scores (Tensor): Classification outputs
                of each decoder layers. Each is a 4D-tensor, has shape
                (num_decoder_layers, bs, num_queries, cls_out_channels).
            all_layers_bbox_preds (Tensor): Sigmoid regression
                outputs of each decoder layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                (num_decoder_layers, bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'

        losses_cls, _, losses_bbox, losses_iou = multi_apply(
            self.loss_by_feat_single,
            all_layers_cls_scores,
            [None for _ in range(len(all_layers_cls_scores))],
            all_layers_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in \
                zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict