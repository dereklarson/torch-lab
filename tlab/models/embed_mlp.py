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


class MLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        n_in = cfg.d_embed * cfg.n_ctx
        n_out = n_in
        layers = []
        for n_out in cfg.mlp_layers:
            layers.append(nn.Linear(n_in, n_out))
            n_in = n_out
        self.layers = nn.ModuleList(layers)
        self.n_out = n_out

    def forward(self, x):
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
        x = self.mlp(torch.flatten(x, 1))
        x = self.unembed(x)
        return x
