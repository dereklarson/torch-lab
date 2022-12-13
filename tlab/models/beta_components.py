from typing import TYPE_CHECKING

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tlab.utils.hookpoint import HookPoint

if TYPE_CHECKING:
    from tlab.models.transformer import TransformerConfig


class EmbeddingAttention(nn.Module):
    def __init__(self, cfg: "TransformerConfig"):
        super().__init__()

        self.W_K = nn.Parameter(
            cfg.weight_alpha
            * torch.randn(cfg.n_heads, cfg.d_head, cfg.n_ctx)
            / np.sqrt(cfg.n_ctx)
        )
        self.W_Q = nn.Parameter(
            cfg.weight_alpha
            * torch.randn(cfg.n_heads, cfg.d_head, cfg.n_ctx)
            / np.sqrt(cfg.n_ctx)
        )
        self.W_V = nn.Parameter(
            cfg.weight_alpha
            * torch.randn(cfg.n_heads, cfg.d_head, cfg.n_ctx)
            / np.sqrt(cfg.n_ctx)
        )
        self.W_O = nn.Parameter(
            cfg.weight_alpha
            * torch.randn(cfg.n_ctx, cfg.d_head * cfg.n_heads)
            / np.sqrt(cfg.n_ctx)
        )
        self.d_head = cfg.d_head
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x):
        k = self.hook_k(torch.einsum("ahc,bce->baeh", self.W_K, x))
        q = self.hook_q(torch.einsum("ahc,bce->baeh", self.W_Q, x))
        v = self.hook_v(torch.einsum("ahc,bce->baeh", self.W_V, x))
        attn_scores_pre = torch.einsum("baeh,baEh->baeE", k, q)
        attn_matrix = self.hook_attn(
            F.softmax(
                self.hook_attn_pre(attn_scores_pre / np.sqrt(self.d_head)),
                dim=-1,
            )
        )
        z = self.hook_z(torch.einsum("baeh,baeE->baEh", v, attn_matrix))
        z_flat = einops.rearrange(z, "b a e h -> b e (a h)")
        out = torch.einsum("cf,bef->bce", self.W_O, z_flat)
        return out
