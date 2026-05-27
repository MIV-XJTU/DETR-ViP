import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS

@MODELS.register_module()
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, loss_weight=1.0, normalize=False):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.normalize = normalize

    def forward(self, visual_prompts, text_prompts, targets):
        if self.normalize:
            visual_prompts = F.normalize(visual_prompts, p=2, dim=-1)
            text_prompts = F.normalize(text_prompts, p=2, dim=-1)
        similarity = (visual_prompts @ text_prompts.transpose(-1, -2)) / self.temperature
        targets = targets.float() / targets.sum(-1).unsqueeze(-1)
        loss_align = -targets * F.log_softmax(similarity, dim=-1)
        return loss_align.sum(-1) * self.loss_weight