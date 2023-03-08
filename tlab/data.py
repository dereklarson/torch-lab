"""Methods for generating synthetic data.
"""
import itertools
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
import torch

OPERATION_MAP = {
    "add": "+",
    "par": "+",
    "sub": "-",
    "mul": "*",
}


@dataclass
class DataConfig:
    value_range: int  # Range of integers involved in the arithmetic
    operation: str  # Numerical operation to use for label calculation
    training_fraction: float  # Amount of generated data put into training set
    data_seed: int  # Numpy RNG seed for generating data (separate from PyTorch)
    dist_style: str = "normal"  # Keyword to trigger different distributions of data
    value_count: int = 2  # How many input values (i.e. Transformer context)
    # 'base' will default to 'value_range' if not set
    base: Optional[int] = None  # Base of symbols, e.g. base 16 is hexadecimal.
    use_operators: bool = False  # Whether to include the operator tokens
    range_dict: Dict[str, Tuple] = None  # Option to broadcast to multiple configs


class DataDiv:
    def __init__(
        self, inputs: List[Tuple[int, ...]], labels: List[int], to_cuda: bool = True
    ) -> None:
        self.inputs = inputs
        self.labels = labels
        if to_cuda:
            self.inputs = torch.tensor(inputs).to("cuda")
            self.labels = torch.tensor(labels).to("cuda")


class Dataset:
    def __init__(
        self,
        cfgs: List[DataConfig],
        vocabulary: List[str],
        train: DataDiv,
        test: DataDiv,
    ) -> None:
        self.cfgs = cfgs
        self.vocabulary = vocabulary
        self.train = train
        self.test = test

    @classmethod
    def from_config(cls, main_cfg: DataConfig, to_cuda: bool = True) -> "Dataset":
        cfgs = cls._expand_params(main_cfg)
        vocabulary = cls.create_vocabulary(main_cfg)
        use_operators = any(cfg.use_operators for cfg in cfgs)

        train_inputs, train_labels = [], []
        test_inputs, test_labels = [], []
        np.random.seed(main_cfg.data_seed)
        for cfg in cfgs:
            generator = _asymm if cfg.dist_style == "asymm" else _uniform
            curr_train_in, curr_test_in = generator(cfg)

            # Apply label first, which is easier before inserting operator tokens
            operation = _get_operation(cfg)
            train_labels.extend(np.apply_along_axis(operation, 1, curr_train_in))
            test_labels.extend(np.apply_along_axis(operation, 1, curr_test_in))
            if use_operators:
                op_token_idx = vocabulary.index(OPERATION_MAP.get(cfg.operation, "XX"))
                _insert_op(curr_train_in, op_token_idx)
                _insert_op(curr_test_in, op_token_idx)
            train_inputs.extend(curr_train_in)
            test_inputs.extend(curr_test_in)

        if use_operators:
            eq_token_idx = vocabulary.index("=")
            _append_eq(train_inputs, eq_token_idx)
            _append_eq(test_inputs, eq_token_idx)

        train = DataDiv(train_inputs, train_labels, to_cuda)
        test = DataDiv(test_inputs, test_labels, to_cuda)
        return Dataset(cfgs, vocabulary, train, test)

    @classmethod
    def _expand_params(cls, cfg: DataConfig) -> List[DataConfig]:
        if not cfg.range_dict:
            return [cfg]
        assert len(cfg.range_dict) == 1, "Only single ranges in DataConfig for now"
        cfgs = []
        for key, val_range in cfg.range_dict.items():
            for val in val_range:
                base = asdict(cfg)
                base[key] = val
                cfgs.append(DataConfig(**base))
        return cfgs

    @classmethod
    def create_vocabulary(cls, main_cfg: DataConfig, **kwargs) -> List[str]:
        cfgs = cls._expand_params(main_cfg)
        bases = {cfg.base or cfg.value_range for cfg in cfgs}
        assert len(bases) == 1, "DataConfigs have unequal bases"
        base = bases.pop()

        vocab = list(map(str, range(base)))

        if any(cfg.use_operators for cfg in cfgs):
            for cfg in cfgs:
                vocab.append(OPERATION_MAP.get(cfg.operation, "XX"))
            vocab.append("=")

        return vocab


def _get_operation(cfg: DataConfig):
    if cfg.operation == "add":
        operation = lambda x: sum(x) % cfg.value_range
    elif cfg.operation == "sub":
        operation = lambda x: (x[0] - sum(x[1:])) % cfg.value_range
    elif cfg.operation == "mul":
        operation = lambda x: np.prod(x) % cfg.value_range
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
