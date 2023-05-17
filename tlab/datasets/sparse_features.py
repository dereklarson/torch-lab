"""Methods for generating synthetic data.
"""
import itertools
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from tlab.datasets.lab_dataset import DataBatch, LabDataset
from tlab.utils.util import to_numpy


class SparseFeatures(LabDataset):
    @dataclass
    class Config(LabDataset.Config):
        data_size: int = 1000
        n_features: int = 10000
        p_active: float = 0.001

    def __init__(
        self,
        cfg: Config,
    ) -> None:
        self.config = cfg

        feat = torch.rand((cfg.data_size, cfg.n_features), device="cuda")
        self.train = torch.nn.functional.normalize(
            torch.where(
                torch.rand((cfg.data_size, cfg.n_features), device="cuda")
                <= cfg.p_active,
                feat,
                torch.zeros((), device="cuda"),
            )
        )
        self.val = torch.nn.functional.normalize(
            torch.where(
                torch.rand((cfg.data_size, cfg.n_features), device="cuda")
                <= cfg.p_active,
                feat,
                torch.zeros((), device="cuda"),
            )
        )

    def get_batch(self, split: str) -> DataBatch:
        if split == "train":
            return DataBatch(self.train, self.train)
        else:
            return DataBatch(self.val, self.val)

    @property
    def val_loader(self):
        return [self.get_batch("val")]
