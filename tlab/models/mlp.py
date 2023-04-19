"""Simple MLP
"""
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tlab.models.beta_components import MixLayer, MultLayer, SimpleMultLayer
from tlab.models.components import Embed, LinearLayer, Unembed
from tlab.models.lab_model import LabModel
from tlab.utils.hookpoint import HookPoint


class MLP(LabModel):
    @dataclass
    class Config(LabModel.Config):
        n_inputs: int
        n_outputs: int
        mlp_layers: Tuple[int, ...]
        use_bias: bool = True
        use_dropout: bool = False
        layer_type: str = "Linear"
        activation_type: str = "ReLU"
        init_method: str = "kaiming"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        n_in = cfg.n_inputs
        layers = []
        for n_out in cfg.mlp_layers:
            if cfg.layer_type == "Mix":
                layers.append(MixLayer(n_in, n_out, cfg.use_bias))
            elif cfg.layer_type == "Mult":
                layers.append(MultLayer(n_in, n_out, cfg.use_bias))
            elif cfg.layer_type == "SimpleMult":
                layers.append(SimpleMultLayer(n_in, n_out))
            elif cfg.layer_type == "Linear":
                layers.append(LinearLayer(n_in, n_out, cfg.init_method, cfg.use_bias))
            else:
                raise Exception(f"No layer for type '{cfg.layer_type}'")
            n_in = n_out
        self.layers = nn.ModuleList(layers)
        self.output = Unembed(n_in, cfg.n_outputs)

        for idx in range(len(cfg.mlp_layers)):
            setattr(self, f"hook_layer.{idx}", HookPoint())
        self._init_hooks()

    def forward(self, x):
        x = torch.flatten(x, 1)
        for idx, layer in enumerate(self.layers):
            if self.config.activation_type == "ReLU":
                x = F.relu(layer(x))
            elif self.config.activation_type == "LeakyReLU":
                x = nn.LeakyReLU(0.01)(layer(x))
            x = getattr(self, f"hook_layer.{idx}")(x)
        if self.config.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.output(x)
        return x


class EmbedMLP(LabModel):
    @dataclass
    class Config(LabModel.Config):
        d_embed: int
        mlp_layers: List[int]
        n_vocab: int
        n_ctx: int
        n_outputs: int
        use_bias: bool = True
        layer_type: str = "Linear"
        activation_type: str = "ReLU"
        init_method: str = "kaiming"

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.embed = Embed(n_vocab=cfg.n_vocab, d_embed=cfg.d_embed)
        mlp_config = MLP.Config.from_parent(cfg, n_inputs=cfg.n_ctx * cfg.d_embed)
        self.mlp = MLP(mlp_config)

        self.hook_embed = HookPoint()
        self._init_hooks()

    def forward(self, x):
        x = self.hook_embed(self.embed(x))
        x = self.mlp(x)
        return x
