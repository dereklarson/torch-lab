"""
A UNet with contextual embedding based on:
https://github.com/TeaPearce/Conditional_Diffusion_MNIST

Intended to be trained by DDPM, see ./diffusion.py
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from tlab.datasets.dataset import DataBatch
from tlab.models.lab_model import LabModel


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        # this adds on correct residual in case channels have increased
        if self.same_channels:
            out = x + x2
        else:
            out = x1 + x2
        return out / 1.414


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(LabModel):
    @dataclass
    class Config(LabModel.Config):
        n_features: int = 64
        input_channels: int = 1
        n_classes: int = 10

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.init_conv = ResidualConvBlock(cfg.input_channels, cfg.n_features)

        self.down1 = UnetDown(cfg.n_features, cfg.n_features)
        self.down2 = UnetDown(cfg.n_features, 2 * cfg.n_features)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * cfg.n_features)
        self.timeembed2 = EmbedFC(1, 1 * cfg.n_features)
        self.contextembed1 = EmbedFC(cfg.n_classes, 2 * cfg.n_features)
        self.contextembed2 = EmbedFC(cfg.n_classes, 1 * cfg.n_features)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * cfg.n_features, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(
                2 * cfg.n_features, 2 * cfg.n_features, 7, 7
            ),  # otherwise just have 2*cfg.n_features
            nn.GroupNorm(8, 2 * cfg.n_features),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * cfg.n_features, cfg.n_features)
        self.up2 = UnetUp(2 * cfg.n_features, cfg.n_features)
        self.out = nn.Sequential(
            nn.Conv2d(2 * cfg.n_features, cfg.n_features, 3, 1, 1),
            nn.GroupNorm(8, cfg.n_features),
            nn.ReLU(),
            nn.Conv2d(cfg.n_features, cfg.input_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.config.n_classes).type(
            torch.float
        )

        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, self.config.n_classes)
        context_mask = -1 * (1 - context_mask)  # need to flip 0 <-> 1
        c = c * context_mask

        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.config.n_features * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.config.n_features * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.config.n_features, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.config.n_features, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out
