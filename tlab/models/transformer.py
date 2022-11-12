"""Transformer implementation (Decoder only) with Hookpoints.

This implementation derives from Neel Nanda's work on Grokking.

Note: I haven't been using the Hookpoints, just left them in place as they should
be useful down the road.

Notes on Einstein notation conventions
 - Repeated, uncontracted indices use caps. e.g. "ij,Ij -> iI" for matrix mult.
Glossary of indices:
 - a: attention head
 - b: batch
 - c: context
 - e: embedding dimension
 - h: head dimension (used in each of k, q, v)
 - m: MLP hidden dimension
"""
from dataclasses import dataclass

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tlab.utils.hookpoint import HookPoint


@dataclass
class TransformerConfig:
    d_embed: int
    d_head: int
    d_mlp: int
    n_ctx: int
    n_heads: int
    n_layers: int
    n_vocab: int
    weight_alpha: float


class Embed(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.W_E = nn.Parameter(
            cfg.weight_alpha
            * torch.randn(cfg.n_vocab, cfg.d_embed)
            / np.sqrt(cfg.d_embed)
        )

    def forward(self, x):
        return self.W_E[x]


class Unembed(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.W_U = nn.Parameter(
            cfg.weight_alpha
            * torch.randn(cfg.d_embed, cfg.n_vocab)
            / np.sqrt(cfg.n_vocab)
        )

    def forward(self, x):
        return x @ self.W_U


class PositionEmbed(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.W_pos = nn.Parameter(
            cfg.weight_alpha
            * torch.randn(cfg.n_ctx, cfg.d_embed)
            / np.sqrt(cfg.d_embed)
        )

    def forward(self, x):
        return x + self.W_pos


# Attention
class Attention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.W_K = nn.Parameter(
            cfg.weight_alpha
            * torch.randn(cfg.n_heads, cfg.d_head, cfg.d_embed)
            / np.sqrt(cfg.d_embed)
        )
        self.W_Q = nn.Parameter(
            cfg.weight_alpha
            * torch.randn(cfg.n_heads, cfg.d_head, cfg.d_embed)
            / np.sqrt(cfg.d_embed)
        )
        self.W_V = nn.Parameter(
            cfg.weight_alpha
            * torch.randn(cfg.n_heads, cfg.d_head, cfg.d_embed)
            / np.sqrt(cfg.d_embed)
        )
        self.W_O = nn.Parameter(
            cfg.weight_alpha
            * torch.randn(cfg.d_embed, cfg.d_head * cfg.n_heads)
            / np.sqrt(cfg.d_embed)
        )
        self.register_buffer("mask", torch.tril(torch.ones((cfg.n_ctx, cfg.n_ctx))))
        self.d_head = cfg.d_head
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x):
        k = self.hook_k(torch.einsum("ahe,bce->bach", self.W_K, x))
        q = self.hook_q(torch.einsum("ahe,bce->bach", self.W_Q, x))
        v = self.hook_v(torch.einsum("ahe,bce->bach", self.W_V, x))
        attn_scores_pre = torch.einsum("bach,baCh->bacC", k, q)
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (
            1 - self.mask[: x.shape[-2], : x.shape[-2]]
        )
        attn_matrix = self.hook_attn(
            F.softmax(
                self.hook_attn_pre(attn_scores_masked / np.sqrt(self.d_head)),
                dim=-1,
            )
        )
        z = self.hook_z(torch.einsum("bach,bacC->baCh", v, attn_matrix))
        z_flat = einops.rearrange(z, "b a c h -> b c (a h)")
        out = torch.einsum("ef,bcf->bce", self.W_O, z_flat)
        return out


class MLP(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.W_in = nn.Parameter(
            cfg.weight_alpha
            * torch.randn(cfg.d_mlp, cfg.d_embed)
            / np.sqrt(cfg.d_embed)
        )
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.W_out = nn.Parameter(
            cfg.weight_alpha
            * torch.randn(cfg.d_embed, cfg.d_mlp)
            / np.sqrt(cfg.d_embed)
        )
        self.b_out = nn.Parameter(torch.zeros(cfg.d_embed))
        self.act_type = "ReLU"
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    def forward(self, x):
        x = self.hook_pre(torch.einsum("me,bce->bcm", self.W_in, x) + self.b_in)
        if self.act_type == "ReLU":
            x = F.relu(x)
        elif self.act_type == "GeLU":
            x = F.gelu(x)
        x = self.hook_post(x)
        x = torch.einsum("em,bcm->bce", self.W_out, x) + self.b_out
        return x


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.hook_attn_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        if cfg.d_mlp > 0:
            self.mlp = MLP(cfg)
            self.hook_resid_post = HookPoint()
            self.hook_mlp_out = HookPoint()

    def forward(self, x):
        x = self.hook_resid_mid(
            x + self.hook_attn_out(self.attn((self.hook_resid_pre(x))))
        )
        if hasattr(self, "mlp"):
            x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp((x))))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.config = cfg
        self.cache = {}

        self.embed = Embed(cfg)
        self.position_embed = PositionEmbed(cfg)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for i in range(cfg.n_layers)]
        )
        self.unembed = Unembed(cfg)

        for name, module in self.named_modules():
            if type(module) == HookPoint:
                module.give_name(name)

    def forward(self, x):
        x = self.embed(x)
        x = self.position_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.unembed(x)
        return x

    def hook_points(self):
        return [module for name, module in self.named_modules() if "hook" in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks("fwd")
            hp.remove_hooks("bwd")

    @property
    def wnorm(self) -> float:
        return float(
            torch.linalg.norm(
                torch.concat([par.flatten() for par in self.parameters()])
            ).to("cpu")
        )

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()

        def save_hook_back(tensor, name):
            cache[name + "_grad"] = tensor[0].detach()

        for hp in self.hook_points():
            hp.add_hook(save_hook, "fwd")
            if incl_bwd:
                hp.add_hook(save_hook_back, "bwd")
