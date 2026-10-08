"""AST-style encoder trained from scratch, with Hopfield (HAST) or standard Transformer blocks.

Both variants share the patch embedding, CLS/DIST tokens, classification head and optional
contrastive projection head; only the encoder block differs (``model_type``).
"""

from copy import deepcopy

import torch
import torch.nn as nn
from hflayers import Hopfield
from hflayers.transformer import HopfieldEncoderLayer
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Block
from torch.amp import autocast


class LayerScale(nn.Module):
    def __init__(self, dim, init_values):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class HopfieldBlock(nn.Module):
    """Pre-norm block with a Hopfield association layer in place of self-attention."""

    def __init__(self, dim, num_heads, mlp_ratio, init_values, proj_drop, drop_path, scaling, max_update_steps):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        association = Hopfield(
            input_size=dim, hidden_size=dim, output_size=dim, pattern_size=dim,
            num_heads=num_heads, scaling=scaling, update_steps_max=max_update_steps,
            association_activation="relu", batch_first=True)
        self.hopfield_layer = HopfieldEncoderLayer(
            hopfield_association=deepcopy(association), dim_feedforward=int(dim * mlp_ratio),
            dropout=proj_drop, activation="relu")
        self.ls1 = LayerScale(dim, init_values)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(proj_drop),
            nn.Linear(hidden, dim), nn.Dropout(proj_drop))
        self.ls2 = LayerScale(dim, init_values)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.hopfield_layer(self.norm1(x))))
        return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))


class ASTModel(nn.Module):
    def __init__(self, n_class=35, fstride=10, tstride=10, input_fdim=128, input_tdim=128,
                 model_type="hopfield", dim=64, depth=4, num_heads=2, mlp_ratio=3.0,
                 proj_drop=0.15, attn_drop=0.0, drop_path=0.08,
                 hopfield_scaling=1.0, hopfield_max_update_steps=2, hopfield_init_values=0.2,
                 contrastive_lambda=0.0, contrastive_projection_dim=64):
        super().__init__()
        # 16x16 patches with (fstride, tstride) overlap -> f_dim x t_dim patches
        f_dim = (input_fdim - 16) // fstride + 1
        t_dim = (input_tdim - 16) // tstride + 1
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=(16, 16), stride=(fstride, tstride))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, f_dim * t_dim + 2, dim))
        trunc_normal_(self.pos_embed, std=.02)

        if model_type == "hopfield":
            self.blocks = nn.Sequential(*[
                HopfieldBlock(dim, num_heads, mlp_ratio, hopfield_init_values, proj_drop, drop_path,
                              hopfield_scaling, hopfield_max_update_steps)
                for _ in range(depth)])
        elif model_type == "transformer":
            self.blocks = nn.Sequential(*[
                Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                      drop=proj_drop, attn_drop=attn_drop, drop_path=drop_path)
                for _ in range(depth)])
        else:
            raise ValueError(f"model_type must be 'hopfield' or 'transformer', got {model_type!r}")

        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_class))
        self.contrastive_projection = None
        if contrastive_lambda > 0.0:
            self.contrastive_projection = nn.Sequential(
                nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, contrastive_projection_dim))

    def forward_features(self, x):
        """(B, time, freq) spectrogram -> (B, dim): mean of the CLS and DIST tokens."""
        x = self.patch_embed(x.unsqueeze(1).transpose(2, 3)).flatten(2).transpose(1, 2)
        B = x.shape[0]
        x = torch.cat((self.cls_token.expand(B, -1, -1), self.dist_token.expand(B, -1, -1), x), dim=1)
        x = self.blocks(x + self.pos_embed)
        return self.norm(x)[:, :2].mean(dim=1)

    @autocast("cuda")
    def forward(self, x, return_contrastive=False):
        features = self.forward_features(x)
        logits = self.mlp_head(features)
        if return_contrastive:
            return logits, self.contrastive_projection(features)
        return logits
