"""Simple MLP
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tlab.models.beta_components import MixLayer, MultLayer, SimpleMultLayer
from tlab.models.components import LinearLayer, Unembed
from tlab.models.lab_model import LabModel, ModelConfig


class MLP(LabModel):
    @dataclass
    class Config(ModelConfig):
        n_inputs: int
        n_outputs: int
        mlp_layers: List[int]
        use_bias: bool = True
        layer_type: str = "Linear"

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
                layers.append(LinearLayer(n_in, n_out, cfg.use_bias))
            else:
                raise Exception(f"No layer for type '{cfg.layer_type}'")
            n_in = n_out
        self.layers = nn.ModuleList(layers)
        self.output = Unembed(n_in, cfg.n_outputs)

    def forward(self, x):
        x = torch.flatten(x, 1)
        for layer in self.layers:
            x = F.relu(layer(x))
        x = F.dropout(x, training=self.training)
        x = self.output(x)
        return x
