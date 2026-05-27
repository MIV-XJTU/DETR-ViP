import copy

import torch
import torch.nn.functional as F
from torch import nn

from mmdet.registry import MODELS

from transformers import CLIPModel, CLIPProcessor

@MODELS.register_module()
class CLIPTextModel(nn.Module):
    def __init__(self, clip_path, freeze=True):
        super().__init__()
        self.tokenizer = CLIPProcessor.from_pretrained(clip_path)
        clip_model = CLIPModel.from_pretrained(clip_path)
        self.encoder = copy.deepcopy(clip_model.text_model)
        self.prompt_list = [\
            "a photo of a {}", \
            "a clear photo of a {}", \
            "a cropped photo of the {}", \
            "a photo that appears to contain a {}",\
            "a low-resolution photo of the {}"]
        self.language_dim = self.encoder.embeddings.token_embedding.weight.shape[-1]
        if freeze:
            self.freeze_text_encoder()

    def freeze_text_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, text, device=None):
        text_embeddings = []
        for prompt in self.prompt_list:
            inputs = [prompt.format(_) for _ in text]
            inputs = self.tokenizer(text=inputs, return_tensors="pt", padding=True).to(device)
            outputs = self.encoder(**inputs).pooler_output
            text_embeddings.append(outputs)
        text_embeddings = torch.stack(text_embeddings, -1)
        text_embeddings = text_embeddings.mean(-1)

        return text_embeddings
