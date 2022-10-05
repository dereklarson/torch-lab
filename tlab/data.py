from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, TypedDict

import numpy as np


class DataDiv(TypedDict):
    In: List[tuple[int, ...]]
    Label: List[int]


class Dataset(TypedDict):
    Train: DataDiv
    Val: DataDiv


@dataclass
class DataConfig:
    value_range: int
    operation: Callable[[int, int], int]
    n_values: int
    training_fraction: bool
    save_path: Path


def generate_data(
    cfg: DataConfig,
    seed: int = 0,
):
    value_tuples = np.stack(
        np.meshgrid(*([range(cfg.value_range)]) * cfg.n_values), -1
    ).reshape(-1, cfg.n_values)
    np.random.seed(seed)
    np.random.shuffle(value_tuples)
    div = int(cfg.training_fraction * len(value_tuples))
    labels = np.apply_along_axis(cfg.operation, 1, value_tuples)
    return {
        "Train": {"In": value_tuples[:div], "Label": labels[:div]},
        "Val": {"In": value_tuples[div:], "Label": labels[div:]},
    }
