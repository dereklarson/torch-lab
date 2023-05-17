"""Port Karpathy's usage of the Shakespeare dataset"""
import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch

from tlab.datasets.lab_dataset import DataBatch, LabDataset
from tlab.utils.util import to_numpy


class Shakespeare(LabDataset):
    @dataclass
    class Config(LabDataset.Config):
        batch_size: int = 32
        block_size: int = 256
        eval_iters: int = 200

    def __init__(self, cfg: Config):
        self.config = cfg
        data_dir = os.path.join("./data", "shakespeare")
        self.train_data = np.memmap(
            os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
        )
        self.val_data = np.memmap(
            os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r"
        )
        with open("./data/shakespeare/meta.pkl", "rb") as f:
            self.meta = pickle.load(f)

    def get_batch(self, split):
        # data = train_data if split == 'train' else val_data
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(
            len(data) - self.config.block_size, (self.config.batch_size,)
        )
        x = torch.stack(
            [
                torch.from_numpy(
                    (data[i : i + self.config.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + self.config.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to(
            "cuda", non_blocking=True
        )
        return DataBatch(x, y)

    @property
    def val_loader(self):
        for _ in range(self.config.eval_iters):
            yield self.get_batch("val")

    # def stoi(self, char: str) -> int:
    #     return self.meta["stoi"][char]

    # def itos(self, index: int) -> str:
    #     return self.meta["itos"][index]

    def decode(self, context: torch.Tensor) -> str:
        return "".join([self.meta["itos"][i] for i in to_numpy(context)])

    def encode(self, inputs: str) -> torch.Tensor:
        return torch.tensor([self.meta["stoi"][c] for c in inputs])
