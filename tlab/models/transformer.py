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
 - h: head dimension (used in each of k, q, o, v)
 - m: MLP hidden dimension
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tlab.models.beta_components import EmbeddingAttention
from tlab.models.lab_model import LabModel, ModelConfig
from tlab.utils.hookpoint import HookPoint


@dataclass
class TransformerConfig(ModelConfig):
    d_embed: int
    d_head: int
    d_mlp: int
    n_ctx: int
    n_heads: int
    n_blocks: int
    n_vocab: int
    p_dropout: float = 0.0
    weight_alpha: float = 1.0
    use_position: bool = True
    attention_style: str = "normal"


class Embed(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.W_E = nn.Parameter(
            cfg.weight_alpha
            * torch.randn(cfg.n_vocab, cfg.d_embed)
            / np.sqrt(cfg.d_embed)
        )

    def forward(self, x):
        return self.W_E[x, :]


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
        if cfg.use_position:
            self.W_pos = nn.Parameter(
                cfg.weight_alpha
                * torch.randn(cfg.n_ctx, cfg.d_embed)
                / np.sqrt(cfg.d_embed)
            )
        else:
            self.W_pos = nn.Parameter(
                torch.zeros(cfg.n_ctx, cfg.d_embed), requires_grad=False
            )

    def forward(self, x):
        return x + self.W_pos


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
            * torch.randn(cfg.n_heads, cfg.d_embed, cfg.d_head)
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

        self.dropout = nn.Dropout(p=cfg.p_dropout)

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
        z = self.hook_z(torch.einsum("bacC,bach->baCh", attn_matrix, v))
        out = torch.einsum("aeh,bach->bce", self.W_O, z)
        out = self.dropout(out)
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
        if cfg.attention_style == "cross":
            self.attn = EmbeddingAttention(cfg)
        else:
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


class Transformer(LabModel):
    def __init__(self, cfg: TransformerConfig):
        super().__init__(cfg)

        self.embed = Embed(cfg)
        self.position_embed = PositionEmbed(cfg)

        # TODO Consider allowing variants to the standard TransformerBlock
        block = TransformerBlock

        self.blocks = nn.ModuleList([block(cfg) for i in range(cfg.n_blocks)])
        self.unembed = Unembed(cfg)

        self._init_hooks()

    def forward(self, x):
        x = self.embed(x)
        x = self.position_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.unembed(x)
        # Use the last position of context for the prediction
        return x[:, -1]
