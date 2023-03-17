"""Simple MLP
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tlab.models.lab_model import LabModel, ModelConfig


@dataclass
class MLPConfig(ModelConfig):
    d_embed: int
    mlp_layers: List[int]
    n_vocab: int
    n_ctx: int
    n_outputs: int
    use_bias: bool = True
    use_multlayer: bool = False


class Embed(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.W_E = nn.Parameter(
            torch.randn(cfg.n_vocab, cfg.d_embed) / np.sqrt(cfg.d_embed)
        )

    def forward(self, x):
        return self.W_E[x, :]


class Unembed(nn.Module):
    def __init__(self, cfg: MLPConfig, n_in: int):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(n_in, cfg.n_outputs) / np.sqrt(n_in))

    def forward(self, x):
        return x @ self.W_U


class MLPLayer(nn.Module):
    def __init__(self, cfg: MLPConfig, n_in: int, n_out: int):
        super().__init__()
        self.cfg = cfg
        # self.weight = nn.Parameter(torch.rand(n_in, n_out) / np.sqrt(n_in))
        self.weight = nn.Parameter(
            torch.FloatTensor(n_out, n_in).uniform_(-1, 1) / np.sqrt(n_in)
        )
        if self.cfg.use_bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(n_out).uniform_(-1, 1) / np.sqrt(n_in)
            )

    def forward(self, x):
        x = x @ self.weight.T
        if self.cfg.use_bias:
            x += self.bias
        return x


class MultLayer(nn.Module):
    def __init__(self, cfg: MLPConfig, n_in: int, n_out: int):
        super().__init__()
        self.cfg = cfg
        self.W_A = nn.Parameter(
            torch.FloatTensor(n_out, n_in).uniform_(-1, 1) / np.sqrt(n_in)
        )
        self.W_B = nn.Parameter(
            torch.FloatTensor(n_out, n_in).uniform_(-1, 1) / np.sqrt(n_in)
        )
        if self.cfg.use_bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(n_out).uniform_(-1, 1) / np.sqrt(n_in)
            )

    def forward(self, x):
        left = x @ self.W_A.T
        right = x @ self.W_B.T
        x = left * right
        if self.cfg.use_bias:
            x += self.bias
        return x


class MLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        n_in = cfg.d_embed * cfg.n_ctx
        n_out = n_in
        layers = []
        for n_out in cfg.mlp_layers:
            if cfg.use_multlayer:
                layers.append(MultLayer(cfg, n_in, n_out))
            else:
                layers.append(MLPLayer(cfg, n_in, n_out))
            n_in = n_out
        self.layers = nn.ModuleList(layers)
        self.n_out = n_out

    def forward(self, x):
        x = torch.flatten(x, 1)
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


class EmbedMLP(LabModel):
    def __init__(self, cfg: MLPConfig):
        super().__init__(cfg)

        self.embed = Embed(cfg)
        self.mlp = MLP(cfg)
        self.unembed = Unembed(cfg, self.mlp.n_out)

        self._init_hooks()

    def forward(self, x):
        x = self.embed(x)
        x = self.mlp(x)
        x = self.unembed(x)
        return x
