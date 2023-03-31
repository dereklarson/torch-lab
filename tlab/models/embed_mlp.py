"""Simple MLP
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tlab.models.beta_components import MixLayer, MultLayer, SimpleMultLayer
from tlab.models.components import LinearLayer
from tlab.models.lab_model import LabModel, ModelConfig
from tlab.utils.hookpoint import HookPoint


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
        self.W_U = nn.Parameter(torch.randn(cfg.n_outputs, n_in) / np.sqrt(n_in))

    def forward(self, x):
        return x @ self.W_U.T


class MLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        n_in = cfg.d_embed * cfg.n_ctx
        n_out = n_in
        layers = []
        for n_out in cfg.mlp_layers:
            if cfg.layer_type == "Mix":
                layers.append(MixLayer(n_in, n_out, cfg.use_bias))
            elif cfg.layer_type == "Mult":
                layers.append(MultLayer(n_in, n_out, cfg.use_bias))
            elif cfg.layer_type == "SimpleMult":
                layers.append(SimpleMultLayer(n_in, n_out, cfg.use_bias))
            elif cfg.layer_type == "Linear":
                layers.append(LinearLayer(n_in, n_out, cfg.use_bias))
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

        self.hook_embed = HookPoint()
        self.hook_mlp = HookPoint()

        self._init_hooks()

    def forward(self, x):
        x = self.hook_embed(self.embed(x))
        x = self.hook_mlp(self.mlp(x))
        x = self.unembed(x)
        return x
