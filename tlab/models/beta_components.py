import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tlab.models.components import LinearLayer
from tlab.models.lab_model import LabModel
from tlab.utils.hookpoint import HookPoint


class MultLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int, use_bias: bool = True):
        super().__init__()
        self.use_bias = use_bias
        self.W_A = nn.Parameter(
            torch.FloatTensor(n_out, n_in).uniform_(-1, 1) / np.sqrt(n_in)
        )
        self.W_B = nn.Parameter(
            torch.FloatTensor(n_out, n_in).uniform_(-1, 1) / np.sqrt(n_in)
        )
        if self.use_bias:
            self.bias_a = nn.Parameter(
                torch.FloatTensor(n_out).uniform_(-1, 1) / np.sqrt(n_in)
            )
            self.bias_b = nn.Parameter(
                torch.FloatTensor(n_out).uniform_(-1, 1) / np.sqrt(n_in)
            )

    def forward(self, x):
        left = x @ self.W_A.T
        right = x @ self.W_B.T
        if self.use_bias:
            left += self.bias_a
            right += self.bias_b
        x = left * right
        return x


class SimpleMultLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int, use_bias: bool = True):
        super().__init__()
        self.use_bias = use_bias
        self.weight = nn.Parameter(
            torch.FloatTensor(n_out, n_in).uniform_(-1, 1) / np.sqrt(n_in)
        )
        if self.use_bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(n_out).uniform_(-1, 1) / np.sqrt(n_in)
            )

    def forward(self, x):
        coeff = x @ self.weight.T
        if self.use_bias:
            coeff += self.bias
        x = x * coeff
        return x


class EmbeddingAttention(nn.Module):
    def __init__(self, cfg: LabModel.Config):
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
