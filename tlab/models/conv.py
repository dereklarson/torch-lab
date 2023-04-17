from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tlab.models.beta_components import MixLayer
from tlab.models.components import LinearLayer, Unembed
from tlab.models.lab_model import LabModel
from tlab.models.mlp import MLP


class LeNet(LabModel):
    @dataclass
    class Config(LabModel.Config):
        n_inputs: int
        n_outputs: int
        mlp_layers: Tuple[int, ...] = (120, 84)
        use_bias: bool = True
        layer_type: str = "Linear"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        mlp_config = MLP.Config(**{**asdict(cfg), "n_inputs": 16 * 5 * 5})
        self.mlp = MLP(mlp_config)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.mlp(x)
        return x


class ConvNet(LabModel):
    @dataclass
    class Config(LabModel.Config):
        n_outputs: int
        mlp_layer: int
        n_inputs: int = 784
        use_bias: bool = True
        layer_type: str = "Linear"

    def __init__(self, cfg: Config):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        if cfg.layer_type == "Mix":
            self.fc1 = MixLayer(320, cfg.mlp_layer, cfg.use_bias)
        elif cfg.layer_type == "Linear":
            self.fc1 = nn.Linear(320, cfg.mlp_layer, bias=cfg.use_bias)
        else:
            raise Exception(f"No layer for type '{cfg.layer_type}'")
        self.fc2 = nn.Linear(50, cfg.n_outputs)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
