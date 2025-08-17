# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py

from importlib import resources
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

ASSETS_PATH = resources.files("assets")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        state_dict = torch.load(
            ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth")
        )
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)

class DeepFM(nn.Module):
    def __init__(self, feature_sizes, embedding_size=8, hidden_dims=[800, 800, 800], dropout=[0.5, 0.5, 0.5], fusion_module=None, fusion_input_dim=768):
        super(DeepFM, self).__init__()
        self.field_size = len(feature_sizes)
        self.embedding_size = embedding_size
        self.fusion_module = fusion_module
        self.fusion_input_dim = fusion_input_dim

        # FM parts
        self.fm_first_order = nn.ModuleList([
            nn.Embedding(size, 1) for size in feature_sizes
        ])
        self.fm_second_order = nn.ModuleList([
            nn.Embedding(size, embedding_size) for size in feature_sizes
        ])

        # Deep part
        # Deep part input size
        base_deep_input_dim = self.field_size * embedding_size
        total_deep_input_dim = base_deep_input_dim + (fusion_module.output_dim if fusion_module else 0)

        all_dims = [total_deep_input_dim] + hidden_dims + [1]
        self.deep_layers = nn.ModuleList([
            nn.Linear(all_dims[i], all_dims[i+1]) for i in range(len(all_dims)-1)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(p) for p in dropout
        ])
        self.relu = nn.ReLU()
        self.bias = nn.Parameter(torch.zeros(1))

    def cache_embeddings(self, text_embed, image_embed):
        self._cached_text_embed = text_embed
        self._cached_image_embed = image_embed

    def forward(self, Xi, Xv):
        # First-order term
        first_order = [emb(Xi[:, i]) * Xv[:, i].unsqueeze(1) for i, emb in enumerate(self.fm_first_order)]
        first_order_sum = torch.sum(torch.cat(first_order, dim=1), dim=1)

        # Second-order term
        embeds = [emb(Xi[:, i]) * Xv[:, i].unsqueeze(1) for i, emb in enumerate(self.fm_second_order)]
        summed = sum(embeds)
        summed_square = summed * summed
        square_sum = sum([x * x for x in embeds])
        second_order_sum = 0.5 * torch.sum(summed_square - square_sum, dim=1)

        # Deep input from FM embeddings
        deep_input = torch.cat(embeds, dim=1)

        # Optional: add fusion output to deep input
        if self.fusion_module is not None:
            gender_idx = Xi[:, 2]  # gender index
            age_idx = Xi[:, 3]     # age index
            #fusion_input = Xv[:, -self.fusion_input_dim:]         # shape (batch, 1536)
            #half = self.fusion_input_dim // 2                     # 768
            #text_embed = fusion_input[:, :half]                   # (batch, 768)
            #image_embed = fusion_input[:, half:]                  # (batch, 768)
            
            fusion_out = self.fusion_module(self._cached_text_embed, self._cached_image_embed, gender_idx, age_idx)
            deep_input = torch.cat([deep_input, fusion_out], dim=1)

        # Deep forward pass
        for i, layer in enumerate(self.deep_layers[:-1]):
            deep_input = self.relu(layer(deep_input))
            if i < len(self.dropouts):
                deep_input = self.dropouts[i](deep_input)
        deep_output = self.deep_layers[-1](deep_input).squeeze(1)

        # Final output
        return first_order_sum + second_order_sum + deep_output + self.bias