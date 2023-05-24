"""Methods for generating synthetic data.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from tlab.datasets.lab_dataset import DataBatch, LabDataset


class SparseFeatures(LabDataset):
    @dataclass
    class Config(LabDataset.Config):
        train_samples: int = 1000
        n_features: int = 10000
        p_active: float = 0.001

    def __init__(
        self,
        cfg: Config,
    ) -> None:
        super().__init__(cfg)

        full_feat = torch.tensor(np.random.rand(cfg.train_samples * 2, cfg.n_features))
        mask = torch.tensor(np.random.rand(cfg.train_samples * 2, cfg.n_features))
        features = torch.where(mask <= cfg.p_active, full_feat, torch.zeros(()))
        self.train = F.normalize(features[: cfg.train_samples]).to(cfg.device)
        self.val = F.normalize(features[cfg.train_samples :]).to(cfg.device)

    def get_batch(self, split: str) -> DataBatch:
        if split == "train":
            return DataBatch(self.train, self.train)
        else:
            return DataBatch(self.val, self.val)

    @property
    def val_loader(self):
        return [self.get_batch("val")]
