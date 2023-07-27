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
        total_samples: int = 2000
        n_features: int = 10000
        p_active: float = 0.001
        use_choice: bool = False
        const_mag: bool = False
        feat_epsilon: float = 0.01

    def __init__(
        self,
        cfg: Config,
    ) -> None:
        super().__init__(cfg)

        if cfg.const_mag:
            full_feat = 0.5 + cfg.feat_epsilon * np.random.rand(
                1, cfg.n_features
            ).repeat(cfg.total_samples, 0)
        else:
            full_feat = np.random.rand(cfg.total_samples, cfg.n_features)

        if cfg.use_choice:
            count = int(cfg.n_features * cfg.p_active)
            if count < cfg.n_features * cfg.p_active:
                print(f"Warning: use_choice=True with fractional feature activation.")
            mask = np.ones((cfg.total_samples, cfg.n_features))
            for idx in range(cfg.total_samples):
                active_idxs = np.random.choice(cfg.n_features, count, replace=False)
                mask[idx, active_idxs] = 0
        else:
            mask = np.random.rand(cfg.total_samples, cfg.n_features)

        full_feat = torch.tensor(full_feat, dtype=torch.float).to(cfg.device)
        mask = torch.tensor(mask, dtype=torch.float).to(cfg.device)

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


class RingFeatures(LabDataset):
    @dataclass
    class Config(LabDataset.Config):
        train_samples: int = 100
        n_features: int = 1000
        n_active: float = 10
        n_overlap: int = 0

    def __init__(
        self,
        cfg: Config,
    ) -> None:
        super().__init__(cfg)

        train_feat = torch.zeros((cfg.train_samples, cfg.n_features))
        val_feat = torch.zeros((cfg.train_samples, cfg.n_features))

        for idx in range(cfg.train_samples):
            start = (idx * (cfg.n_active - cfg.n_overlap)) % cfg.n_features
            end = start + cfg.n_active
            if end >= cfg.n_features:
                train_feat[idx, start:] = 1
                train_feat[idx, : (end - cfg.n_features)] = 1
            else:
                train_feat[idx, start:end] = 1

            val_idxs = np.random.choice(cfg.n_features, cfg.n_active, replace=False)
            val_feat[idx, val_idxs] = 1

        self.train = F.normalize(train_feat).to(cfg.device)
        self.val = F.normalize(val_feat).to(cfg.device)

    def get_batch(self, split: str) -> DataBatch:
        if split == "train":
            return DataBatch(self.train, self.train)
        else:
            return DataBatch(self.val, self.val)

    @property
    def val_loader(self):
        return [self.get_batch("val")]


class AssignedFeatures(LabDataset):
    """In this dataset, features are only activated a strict number of times."""

    @dataclass
    class Config(LabDataset.Config):
        train_samples: int = 100
        n_features: int = 1000
        n_active: float = 10

    def __init__(
        self,
        cfg: Config,
    ) -> None:
        super().__init__(cfg)

        train_feat = torch.zeros((cfg.train_samples, cfg.n_features))
        val_feat = torch.zeros((cfg.train_samples, cfg.n_features))

        max_feat_activation = int(
            np.ceil(cfg.train_samples * cfg.n_active / cfg.n_features)
        )
        assert cfg.n_features % cfg.n_active == 0, "'n_active' % 'n_features' must be 0"
        ts_per_group = cfg.n_features // cfg.n_active
        sample_idx = 0
        for _ in range(max_feat_activation):
            idxs = np.random.permutation(cfg.n_features)
            val_idxs = np.random.permutation(cfg.n_features)
            for f_idx in range(ts_per_group):
                start = f_idx * cfg.n_active
                end = (f_idx + 1) * cfg.n_active
                train_row = torch.FloatTensor(np.random.rand(1, cfg.n_active))
                train_feat[sample_idx, idxs[start:end]] = train_row
                val_row = torch.FloatTensor(np.random.rand(1, cfg.n_active))
                val_feat[sample_idx, val_idxs[start:end]] = val_row
                sample_idx += 1
                if sample_idx >= cfg.train_samples:
                    break

        self.train = F.normalize(train_feat).to(cfg.device)
        self.val = F.normalize(val_feat).to(cfg.device)

    def get_batch(self, split: str) -> DataBatch:
        if split == "train":
            return DataBatch(self.train, self.train)
        else:
            return DataBatch(self.val, self.val)

    @property
    def val_loader(self):
        return [self.get_batch("val")]
