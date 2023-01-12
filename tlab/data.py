"""Methods for generating synthetic data.
"""
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np


class DataDiv(TypedDict):
    In: List[Tuple[int, ...]]
    Label: List[int]


class Dataset(TypedDict):
    Train: DataDiv
    Test: DataDiv


@dataclass
class DataConfig:
    value_range: int  # Range of integers involved in the arithmetic
    operation: str  # Numerical operation to use for label calculation
    training_fraction: float  # Amount of generated data put into training set
    seed: int  # Numpy RNG seed for generating data (separate from PyTorch)
    dist_style: str = "normal"  # Keyword to trigger different distributions of data
    value_count: int = 2  # How many input values (i.e. Transformer context)
    base: Optional[int] = None  # Base of symbols, e.g. base 16 is hexadecimal.
    # 'base' will default to 'value_range' if not set
    use_operators: bool = False  # Whether to include the operator tokens


op_map = {
    "add": "+",
    "par": "+",
    "sub": "-",
    "mul": "*",
}


def create_vocabulary(cfgs: List[DataConfig], **kwargs) -> List[str]:
    bases = {cfg.base or cfg.value_range for cfg in cfgs}
    assert len(bases) == 1, "DataConfigs have unequal bases"
    base = bases.pop()

    vocab = list(map(str, range(base)))

    if any(cfg.use_operators for cfg in cfgs):
        for cfg in cfgs:
            vocab.append(op_map.get(cfg.operation, "XX"))
        vocab.append("=")

    return vocab


def generate_data(cfgs: List[DataConfig], vocabulary: List[str], **kwargs):
    data = {
        "Train": {"In": [], "Label": []},
        "Test": {"In": [], "Label": []},
    }
    use_operators = any(cfg.use_operators for cfg in cfgs)

    for cfg in cfgs:
        np.random.seed(kwargs.get("seed") or cfg.seed)
        generator = _asymm if cfg.dist_style == "asymm" else _uniform

        train, test = generator(cfg)
        # Apply label first, which is easier before inserting operator tokens
        operation = _get_operation(cfg)
        data["Train"]["Label"].extend(np.apply_along_axis(operation, 1, train))
        data["Test"]["Label"].extend(np.apply_along_axis(operation, 1, test))
        if use_operators:
            op_token_idx = vocabulary.index(op_map.get(cfg.operation, "XX"))
            _insert_op(train, op_token_idx)
            _insert_op(test, op_token_idx)
        data["Train"]["In"].extend(train)
        data["Test"]["In"].extend(test)

    if use_operators:
        eq_token_idx = vocabulary.index("=")
        _append_eq(data["Train"]["In"], eq_token_idx)
        _append_eq(data["Test"]["In"], eq_token_idx)

    return data


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


def _insert_op(data: List[List[int]], op_token_idx: int) -> List[List[int]]:
    for row in data:
        for i in range(len(row) - 1, 0, -1):
            row.insert(i, op_token_idx)


def _append_eq(data: List[List[int]], eq_token_idx: int) -> List[List[int]]:
    for row in data:
        row.append(eq_token_idx)


def _uniform(cfg: DataConfig) -> Tuple[List[List[int]], List[List[int]]]:
    """Return all tuples of integers up to a maximum, shuffled and split into train/test."""
    vocab = set(range(cfg.value_range))
    examples = list(map(list, itertools.product(vocab, repeat=cfg.value_count)))
    np.random.shuffle(examples)
    div = int(cfg.training_fraction * len(examples))
    return examples[:div], examples[div:]


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
