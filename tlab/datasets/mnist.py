import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch
import torchvision
from torch.utils.data.dataloader import default_collate

from tlab.datasets.lab_dataset import DataBatch, LabDataset
from tlab.utils.util import to_numpy


class MNIST(LabDataset):
    @dataclass
    class Config(LabDataset.Config):
        batch_size: int = 1000

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.train_data = torchvision.datasets.MNIST(
            "/files/",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )

        self.val_data = torchvision.datasets.MNIST(
            "/files/",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )

        collate_fn = lambda x: tuple(x_.to("cuda") for x_ in default_collate(x))
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        self.validation_loader = torch.utils.data.DataLoader(
            self.val_data,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def get_batch(self, split: str):
        if split == "train":
            return DataBatch(*next(iter(self.train_loader)))
        else:
            return DataBatch(*next(iter(self.val_loader)))

    @property
    def val_loader(self):
        for inputs, targets in self.validation_loader:
            yield DataBatch(inputs, targets)
