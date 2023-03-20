"""Simple MLP
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tlab.models.beta_components import MixLayer, MultLayer
from tlab.models.lab_model import LabModel, ModelConfig


@dataclass
class MLPConfig(ModelConfig):
    d_embed: int
    mlp_layers: List[int]
    n_vocab: int
    n_ctx: int
    n_outputs: int
    use_bias: bool = True
    layer_type: str = "Linear"


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


class LinearLayer(nn.Module):
    def __init__(self, cfg: MLPConfig, n_in: int, n_out: int):
        super().__init__()
        self.cfg = cfg
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


class MLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        n_in = cfg.d_embed * cfg.n_ctx
        n_out = n_in
        layers = []
        for n_out in cfg.mlp_layers:
            if cfg.layer_type == "Mix":
                layers.append(MixLayer(cfg, n_in, n_out))
            elif cfg.layer_type == "Mult":
                layers.append(MultLayer(cfg, n_in, n_out))
            elif cfg.layer_type == "Linear":
                layers.append(LinearLayer(cfg, n_in, n_out))
            else:
                raise Exception(f"No layer for type '{cfg.layer_type}'")
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
