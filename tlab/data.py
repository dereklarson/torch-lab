from dataclasses import dataclass
from typing import List, Tuple, TypedDict

import numpy as np


class DataDiv(TypedDict):
    In: List[Tuple[int, ...]]
    Label: List[int]


class Dataset(TypedDict):
    Train: DataDiv
    Test: DataDiv


@dataclass
class DataConfig:
    style: float
    value_range: int
    operation: str
    training_fraction: bool
    seed: int


def generate_data(cfg: DataConfig, **kwargs):
    np.random.seed(kwargs.get("seed") or cfg.seed)
    generator = _asymm if cfg.style == "asymm" else _uniform
    train, test = generator(cfg)
    operation = _get_operation(cfg)
    return {
        "Train": {"In": train, "Label": np.apply_along_axis(operation, 1, train)},
        "Test": {"In": test, "Label": np.apply_along_axis(operation, 1, test)},
    }


def _get_operation(cfg: DataConfig):
    if cfg.operation == "add":
        operation = lambda x: (x[0] + x[1]) % cfg.value_range
    elif cfg.operation == "sub":
        operation = lambda x: (x[0] - x[1]) % cfg.value_range
    elif cfg.operation == "mul":
        operation = lambda x: (x[0] * x[1]) % cfg.value_range
    else:
        raise Exception(f"Unsupported data operation {cfg.operation}")
    return operation


def _uniform(cfg: DataConfig) -> Tuple[List, List]:
    """Return all pairs of integers up to a maximum, shuffled and split into train/test."""
    pairs = [(i, j) for i in range(cfg.value_range) for j in range(cfg.value_range)]
    np.random.shuffle(pairs)
    div = int(cfg.training_fraction * len(pairs))
    return pairs[:div], pairs[div:]


def _asymm(cfg: DataConfig) -> Tuple[List, List]:
    """Return all pairs of integers (i, j) up to a maximum. Train group has all j >= i."""
    train = [(i, j) for i in range(cfg.value_range) for j in range(i, cfg.value_range)]
    test = [(i, j) for i in range(cfg.value_range) for j in range(0, i)]
    np.random.shuffle(train)
    np.random.shuffle(test)
    # Determine how much of the j < i set we'll add to the train set
    div = int(cfg.training_fraction * len(test))
    train += test[:div]
    test = test[div:]
    return train, test
