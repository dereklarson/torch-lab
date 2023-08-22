"""A simple encoder model inspired by:
https://transformer-circuits.pub/2023/toy-double-descent/index.html
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tlab.models.lab_model import LabModel
from tlab.utils.hookpoint import HookPoint


class SimpleEncoder(LabModel):
    @dataclass
    class Config(LabModel.Config):
        n_inputs: int
        neurons: int
        use_bias: bool = True
        activation_type: str = "ReLU"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.use_bias = cfg.use_bias
        self.weight = nn.Parameter(
            torch.randn(cfg.neurons, cfg.n_inputs) / np.sqrt(cfg.n_inputs)
        )
        self.output = nn.Parameter(
            torch.randn(cfg.neurons, cfg.n_inputs) / np.sqrt(cfg.n_inputs)
        )

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(cfg.neurons))

        self.hook_embed = HookPoint()
        self._init_hooks()

    def forward(self, x):
        x = x @ self.weight.T
        x = self.hook_embed(x)
        if self.use_bias:
            x += self.bias
        if self.config.activation_type == "ReLU":
            x = F.relu(x)
        elif self.config.activation_type == "LeakyReLU":
            x = nn.LeakyReLU(0.01)(x)
        elif self.config.activation_type == "GeLU":
            x = F.gelu(x)
        else:
            raise NotImplementedError(
                f"No activation called {self.config.activation_type}"
            )
        x = x @ F.normalize(self.output, dim=1)
        return x
