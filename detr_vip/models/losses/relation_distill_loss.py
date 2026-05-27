import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS

@MODELS.register_module()
class RelationDistillLoss(nn.Module):
    def __init__(self, temperature_v=0.1, temperature_t=0.07, loss_weight=1.0, normalize=False):
        super(RelationDistillLoss, self).__init__()
        self.temperature_v = temperature_v
        self.temperature_t = temperature_t
        self.loss_weight = loss_weight
        self.normalize = normalize

    def forward_infonce(self, visual_prompts, targets):
        if self.normalize:
            visual_prompts = F.normalize(visual_prompts, p=2, dim=-1)
        similarity = (visual_prompts @ visual_prompts.transpose(-1, -2)) / self.temperature_v
        targets = targets.float() / targets.sum(-1).unsqueeze(-1)
        loss_align = -targets * F.log_softmax(similarity, dim=-1)
        return loss_align.sum(-1) * self.loss_weight

    def forward(self, visual_prompts, text_prompts):
        text_prompts = F.normalize(text_prompts, p=2, dim=-1)
        softlabel = ((text_prompts @ text_prompts.T) / self.temperature_t).softmax(-1)
        loss_distill = self.forward_infonce(visual_prompts, softlabel)
        return loss_distill.mean()

