# Copyright (c) MIV-XJTU. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.task_modules.assigners import FocalLossCost
from mmdet.registry import TASK_UTILS

from mmengine.structures import InstanceData

@TASK_UTILS.register_module()
class DETRViPBinaryFocalLossCost(FocalLossCost):
    def _focal_loss_cost(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_queries, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
            torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
        return cls_cost * self.weight

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``scores`` or ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``labels`` or ``mask``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        # gt_instances.text_token_mask is a repeated tensor of the same length
        # of instances. Only gt_instances.text_token_mask[0] is useful
        token_mask = torch.nonzero(
            gt_instances.token_mask[0]).squeeze(-1)
        pred_scores = pred_instances.scores[:, token_mask]
        gt_labels = gt_instances.positive_maps[:, token_mask]
        return self._focal_loss_cost(pred_scores, gt_labels)