"""Base class for dataset management."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch

from tlab.utils import NameRepr
from tlab.utils.util import to_numpy


class LabDataset(metaclass=NameRepr):
    @dataclass
    class Config:
        dataset_class: Type["LabDataset"]
        data_seed: int

    def __init__(
        self,
        cfg: Config,
    ) -> None:
        pass

    def stats(self) -> None:
        print(
            f"{len(self.train)} training examples, {len(self.val)} validation examples"
        )

    def head(self, n: int = 10, subset: str = "train") -> None:
        for input, label in getattr(self, subset).head(n):
            in_str = " ".join([self.vocabulary[t] for t in input])
            print(f"{in_str} -> {self.vocabulary[label]}")

    @classmethod
    def from_config(cls, cfg: Config, to_cuda: bool = True) -> "LabDataset":
        pass


class DataBatch:
    def __init__(self, inputs, targets) -> None:
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def to_cuda(self):
        self.inputs = torch.tensor(self.inputs).to("cuda")
        self.targets = torch.tensor(self.targets).to("cuda")
        return self

    def head(self, n: int = 10) -> List[Tuple]:
        output = []
        for idx, row in enumerate(self.inputs[:n]):
            output.append((to_numpy(row), int(self.targets[idx])))
        return output
