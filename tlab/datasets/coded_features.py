"""Methods for generating synthetic data.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from tlab.datasets.lab_dataset import DataBatch, LabDataset


class CodedFeatures(LabDataset):
    @dataclass
    class Config(LabDataset.Config):
        train_samples: int = 10000
        total_samples: int = 12000
        d_sphere: int = 256
        n_ground: int = 512
        n_active: float = 5
        mask_decay: float = 0.99

    def __init__(
        self,
        cfg: Config,
    ) -> None:
        super().__init__(cfg)
        self.ground = torch.nn.functional.normalize(
            torch.randn(cfg.n_ground, cfg.d_sphere), dim=1
        ).cuda()
        data = self.generate(cfg.total_samples)

        self.train = data[: cfg.train_samples]
        self.val = data[cfg.train_samples :]

    def generate(self, n_samples: int):
        cfg = self.config
        p_act = cfg.n_active / cfg.n_ground
        decayed = cfg.mask_decay ** torch.arange(0, cfg.n_ground).repeat(n_samples, 1)
        probs = (1 / torch.mean(decayed)) * decayed * p_act
        mask = torch.bernoulli(probs).cuda()
        weights = torch.Tensor(n_samples, cfg.n_ground).uniform_().cuda()

        return (mask * weights) @ self.ground

    def get_batch(self, split: str) -> DataBatch:
        if split == "train":
            return DataBatch(self.train, self.train)
        elif split == "infinite":
            data = self.generate(self.config.train_samples)
            return DataBatch(data, data)
        else:
            return DataBatch(self.val, self.val)

    @property
    def val_loader(self):
        return [self.get_batch("val")]
