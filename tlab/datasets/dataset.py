"""Base class for dataset management."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch

from tlab.utils import NameRepr
from tlab.utils.util import to_numpy


class Dataset(metaclass=NameRepr):
    @dataclass
    class Config:
        dataset_class: Type["Dataset"]
        data_seed: int

    def __init__(
        self,
        cfg: Config,
    ) -> None:
        pass

    def stats(self) -> None:
        print(f"{len(self.train)} training examples, {len(self.test)} test examples")

    def head(self, n: int = 10, subset: str = "train") -> None:
        for input, label in getattr(self, subset).head(n):
            in_str = " ".join([self.vocabulary[t] for t in input])
            print(f"{in_str} -> {self.vocabulary[label]}")

    @classmethod
    def from_config(cls, cfg: Config, to_cuda: bool = True) -> "Dataset":
        pass
