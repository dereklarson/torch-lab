"""Methods for generating synthetic data.
"""
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
    value_range: int  # 'Base' of symbols, e.g. value_range 16 is 'hex'
    operation: str  # Numerical operation to use for label calculation
    training_fraction: float  # Amount of generated data put into training set
    seed: int  # Numpy RNG seed for generating data (separate from PyTorch)
    style: str = "normal"  # Keyword to trigger different distributions of data
    value_count: int = 2  # How many input values (i.e. Transformer context)


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
        operation = lambda x: sum(x) % cfg.value_range
    elif cfg.operation == "sub":
        operation = lambda x: (x[0] - x[1]) % cfg.value_range
    elif cfg.operation == "mul":
        operation = lambda x: (x[0] * x[1]) % cfg.value_range
    elif cfg.operation == "par":
        operation = lambda x: sum(x) % 2
    elif cfg.operation == "max":
        operation = lambda x: max(x)
    elif cfg.operation == "min":
        operation = lambda x: min(x)
    elif cfg.operation == "rand":
        operation = lambda x: np.random.randint(cfg.value_range)
    else:
        raise Exception(f"Unsupported data operation {cfg.operation}")
    return operation


def _uniform(cfg: DataConfig) -> Tuple[List, List]:
    """Return all tuples of integers up to a maximum, shuffled and split into train/test."""
    tuples = np.stack(
        np.meshgrid(*([range(cfg.value_range)]) * cfg.value_count), -1
    ).reshape(-1, cfg.value_count)
    np.random.shuffle(tuples)
    div = int(cfg.training_fraction * len(tuples))
    return tuples[:div], tuples[div:]


def _asymm(cfg: DataConfig) -> Tuple[List, List]:
    """Return all pairs of integers (i, j) up to a maximum. Train group has all j >= i."""
    assert cfg.value_count == 2, f"Can't use asymm setting for value_count != 2"
    train = [(i, j) for i in range(cfg.value_range) for j in range(i, cfg.value_range)]
    test = [(i, j) for i in range(cfg.value_range) for j in range(0, i)]
    np.random.shuffle(test)
    # Determine how much of the j < i set we'll add to the train set
    div = int(cfg.training_fraction * len(test))
    train += test[:div]
    test = test[div:]
    np.random.shuffle(train)
    return train, test
