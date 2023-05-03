"""Transformer implementation (Decoder only) designed for experimentation.

The goal is to have an easily configurable and understandable model architecture,
that supports observing its internals and connects to tools for standardized 
operations.

This implementation largely follows from the following two sources:
 - Andrey Karpathy's NanoGPT: https://github.com/karpathy/nanoGPT
 - Neel Nanda's TransformerLens: https://github.com/neelnanda-io/TransformerLens

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
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tlab.models.components import Embed, LayerNorm, PositionEmbed, Unembed
from tlab.models.lab_model import LabModel
from tlab.utils.hookpoint import HookPoint


class Transformer(LabModel):
    @dataclass
    class Config(LabModel.Config):
        d_embed: int
        d_head: int
        d_mlp: int
        n_ctx: int
        n_heads: int
        n_blocks: int
        n_vocab: int
        p_dropout: float = 0.0
        pos_embed: str = "learned"
        unembed_type: str = "free"
        use_bias: bool = False
        activation_type: str = "ReLU"
        use_flash: bool = True

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.embed = Embed(n_vocab=cfg.n_vocab, d_embed=cfg.d_embed)
        if cfg.pos_embed == "learned":
            self.position_embed = PositionEmbed(n_ctx=cfg.n_ctx, d_embed=cfg.d_embed)
        elif cfg.pos_embed == "cosine":
            raise NotImplementedError("Cosine positional embedding not implemented")
        else:
            pass

        # TODO Consider allowing variants to the standard TransformerBlock
        block = TransformerBlock
        self.blocks = nn.ModuleList([block(cfg) for i in range(cfg.n_blocks)])

        if cfg.unembed_type != "none":
            self.unembed_ln = LayerNorm(n_dim=cfg.d_embed)
            self.unembed = Unembed(n_in=cfg.d_embed, n_outputs=cfg.n_vocab)
        if cfg.unembed_type == "tied":
            self.embed.weight = self.unembed.weight

        self._init_hooks()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if hasattr(module, "weight") and not isinstance(module, LayerNorm):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.embed(x)
        if hasattr(self, "position_embed"):
            x = self.position_embed(x)
        for block in self.blocks:
            x = block(x)
        if hasattr(self, "unembed"):
            x = self.unembed_ln(x)
            x = self.unembed(x)
        return x

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        """
        Beginning with a prompt, sample from the generated next token logits to produce a token
        to append to the prompt. Repeat this for 'max_new_tokens' iterations.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop the sequence to the maximum context allowed.
            ctx = (
                prompt
                if prompt.size(1) <= self.config.n_ctx
                else prompt[:, -self.config.n_ctx :]
            )
            # Select the logits from the last context position.
            logits = self(ctx)[:, -1, :]
            # We can crop to the top_k options to limit poor-quality results if desired.
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # Sample from the distribution given by the probabilities at given temp.
            probs = F.softmax(logits / temperature, dim=-1)
            ctx_next = torch.multinomial(probs, num_samples=1)
            prompt = torch.cat((ctx, ctx_next), dim=1)

        return prompt


class Attention(nn.Module):
    def __init__(self, cfg: Transformer.Config):
        super().__init__()
        self.config = cfg

        self.W_K = nn.Parameter(
            torch.randn(cfg.n_heads, cfg.d_head, cfg.d_embed) / np.sqrt(cfg.d_embed)
        )
        self.W_Q = nn.Parameter(
            torch.randn(cfg.n_heads, cfg.d_head, cfg.d_embed) / np.sqrt(cfg.d_embed)
        )
        self.W_V = nn.Parameter(
            torch.randn(cfg.n_heads, cfg.d_head, cfg.d_embed) / np.sqrt(cfg.d_embed)
        )
        self.W_O = nn.Parameter(
            torch.randn(cfg.n_heads, cfg.d_embed, cfg.d_head) / np.sqrt(cfg.d_embed)
        )
        self.dropout = nn.Dropout(p=cfg.p_dropout)

        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x):
        q = self.hook_q(torch.einsum("ahe,bce->bach", self.W_Q, x).contiguous())
        k = self.hook_k(torch.einsum("ahe,bce->bach", self.W_K, x).contiguous())
        v = self.hook_v(torch.einsum("ahe,bce->bach", self.W_V, x).contiguous())
        if self.config.use_flash:
            z = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.config.p_dropout if self.training else 0,
                is_causal=True,
            )
            out = torch.einsum("aeh,bach->bce", self.W_O, z)
        else:
            print("TODO Fix this implementation")
            attn_scores_pre = torch.einsum("bach,baCh->bacC", k, q)
            attn_scores_masked = (
                torch.tril(attn_scores_pre) - self.mask[: x.shape[-2], : x.shape[-2]]
            )
            attn_matrix = self.hook_attn(
                F.softmax(
                    self.hook_attn_pre(
                        attn_scores_masked / np.sqrt(self.config.d_head)
                    ),
                    dim=-1,
                )
            )
            z = self.hook_z(torch.einsum("bacC,bach->bach", attn_matrix, v))
            out = torch.einsum("aeh,bach->bce", self.W_O, z)
            out = self.dropout(out)
        return out


class TransformerMLP(nn.Module):
    def __init__(self, cfg: Transformer.Config):
        super().__init__()
        if cfg.activation_type == "ReLU":
            self.activation = F.relu
        elif cfg.activation_type == "GeLU":
            self.activation = F.gelu
        self.dropout = nn.Dropout(p=cfg.p_dropout)

        self.W_in = nn.Parameter(
            torch.randn(cfg.d_mlp, cfg.d_embed) / np.sqrt(cfg.d_embed)
        )
        self.W_out = nn.Parameter(
            torch.randn(cfg.d_embed, cfg.d_mlp) / np.sqrt(cfg.d_embed)
        )

        if cfg.use_bias:
            self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
            self.b_out = nn.Parameter(torch.zeros(cfg.d_embed))
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    def forward(self, x):
        prod = torch.einsum("me,bce->bcm", self.W_in, x)
        if hasattr(self, "b_in"):
            prod = prod + self.b_in
        x = self.activation(prod)
        x = torch.einsum("em,bcm->bce", self.W_out, x)
        if hasattr(self, "b_out"):
            x = x + self.b_out
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Transformer.Config):
        super().__init__()
        self.attn_ln = LayerNorm(cfg.d_embed, bias=cfg.use_bias)
        self.attn = Attention(cfg)
        self.hook_attn_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        if cfg.d_mlp > 0:
            self.mlp_ln = LayerNorm(cfg.d_embed, bias=cfg.use_bias)
            self.mlp = TransformerMLP(cfg)
            self.hook_resid_post = HookPoint()
            self.hook_mlp_out = HookPoint()

    def forward(self, x):
        x = self.attn_ln(x)
        x = self.hook_resid_mid(
            x + self.hook_attn_out(self.attn((self.hook_resid_pre(x))))
        )
        if hasattr(self, "mlp"):
            x = self.mlp_ln(x)
            x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp((x))))
        return x
