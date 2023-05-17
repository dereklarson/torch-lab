"""Methods for generating synthetic data.
"""
import itertools
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from tlab.datasets.lab_dataset import DataBatch, LabDataset
from tlab.utils.util import to_numpy

OPERATION_MAP = {
    "add": "+",
    "sub": "-",
    "mul": "*",
}


class Algorithmic(LabDataset):
    @dataclass
    class Config(LabDataset.Config):
        value_range: int = 10  # Range of integers involved in the arithmetic
        result_mod: int = 10  # Apply modulo to the result
        operation: str = "add"  # Numerical operation to use for label calculation
        training_fraction: float = 0.7  # Amount of generated data put into training set
        dist_style: str = "normal"  # Keyword to trigger different distributions of data
        value_count: int = 2  # How many input values (i.e. Transformer context)
        # 'base' will default to 'value_range' if not set
        base: Optional[int] = None  # Base of symbols, e.g. base 16 is hexadecimal.
        use_operators: bool = False  # Whether to include the operator tokens
        range_dict: Dict[str, Tuple] = None  # Option to broadcast to multiple configs

    def __init__(
        self,
        cfgs: List[Config],
        vocabulary: List[str],
        train: DataBatch,
        val: DataBatch,
    ) -> None:
        self.cfgs = cfgs
        self.vocabulary = vocabulary
        self.train = train
        self.val = val

    def head(self, n: int = 10, subset: str = "train") -> None:
        for input, label in getattr(self, subset).head(n):
            in_str = " ".join([self.vocabulary[t] for t in input])
            print(f"{in_str} -> {self.vocabulary[label]}")

    @classmethod
    def from_config(cls, main_cfg: Config, to_cuda: bool = True) -> "LabDataset":
        cfgs = cls._expand_params(main_cfg)
        vocabulary = cls.create_vocabulary(main_cfg)
        use_operators = any(cfg.use_operators for cfg in cfgs)

        train_inputs, train_targets = [], []
        val_inputs, val_targets = [], []
        np.random.seed(main_cfg.data_seed)
        for cfg in cfgs:
            generator = _asymm if cfg.dist_style == "asymm" else _uniform
            curr_train_in, curr_val_in = generator(cfg)

            # Apply label first, which is easier before inserting operator tokens
            operation = _get_operation(cfg)
            if curr_train_in:
                train_targets.extend(np.apply_along_axis(operation, 1, curr_train_in))
            if curr_val_in:
                val_targets.extend(np.apply_along_axis(operation, 1, curr_val_in))
            if use_operators:
                op_token_idx = vocabulary.index(OPERATION_MAP.get(cfg.operation, "XX"))
                _insert_op(curr_train_in, op_token_idx)
                _insert_op(curr_val_in, op_token_idx)
            train_inputs.extend(curr_train_in)
            val_inputs.extend(curr_val_in)

        if use_operators:
            eq_token_idx = vocabulary.index("=")
            _append_eq(train_inputs, eq_token_idx)
            _append_eq(val_inputs, eq_token_idx)

        train = DataBatch(train_inputs, train_targets).to_cuda()
        val = DataBatch(val_inputs, val_targets).to_cuda()
        return cls(cfgs, vocabulary, train, val)

    def get_batch(self, split: str) -> DataBatch:
        if split == "train":
            return self.train
        else:
            return self.val

    @property
    def val_loader(self):
        return [self.val]

    @classmethod
    def _expand_params(cls, cfg: Config) -> List[Config]:
        if not cfg.range_dict:
            return [cfg]
        assert len(cfg.range_dict) == 1, "Only single ranges in Config for now"
        cfgs = []
        for key, val_range in cfg.range_dict.items():
            for val in val_range:
                base = asdict(cfg)
                base[key] = val
                cfgs.append(cls.Config(**base))
        return cfgs

    @classmethod
    def create_vocabulary(cls, main_cfg: Config, **kwargs) -> List[str]:
        cfgs = cls._expand_params(main_cfg)
        bases = {cfg.base or cfg.value_range for cfg in cfgs}
        assert len(bases) == 1, "Configs have unequal bases"
        base = bases.pop()

        vocab = list(map(str, range(base)))

        if any(cfg.use_operators for cfg in cfgs):
            for cfg in cfgs:
                vocab.append(OPERATION_MAP.get(cfg.operation, "XX"))
            vocab.append("=")

        return vocab

    @classmethod
    def nucleus(cls, main_cfg: Config, **kwargs):
        """Create the samples and targets for input ints smaller than modulus."""
        vocab = set(range(main_cfg.result_mod))
        operation = _get_operation(main_cfg)
        inputs = list(map(list, itertools.product(vocab, repeat=main_cfg.value_count)))
        targets = np.apply_along_axis(operation, 1, inputs)
        return DataBatch(inputs, targets).to_cuda()


def _get_operation(cfg: Algorithmic.Config):
    if cfg.operation == "add":
        operation = lambda x: sum(x) % cfg.result_mod
    elif cfg.operation == "sub":
        operation = lambda x: (x[0] - sum(x[1:])) % cfg.result_mod
    elif cfg.operation == "mul":
        operation = lambda x: np.prod(x) % cfg.result_mod
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


def _uniform(cfg: Algorithmic.Config) -> Tuple[List[List[int]], List[List[int]]]:
    """Return all tuples of integers up to a maximum, shuffled and split into train/val."""
    vocab = set(range(cfg.value_range))
    examples = list(map(list, itertools.product(vocab, repeat=cfg.value_count)))
    np.random.shuffle(examples)
    div = int(cfg.training_fraction * len(examples))
    return examples[:div], examples[div:]


def _asymm(cfg: Algorithmic.Config) -> Tuple[List, List]:
    """Return all pairs of integers (i, j) up to a maximum. Train group has all j >= i."""
    assert cfg.value_count == 2, f"Can't use asymm setting for value_count != 2"
    train = [[i, j] for i in range(cfg.value_range) for j in range(i, cfg.value_range)]
    val = [[i, j] for i in range(cfg.value_range) for j in range(0, i)]
    np.random.shuffle(val)
    # Determine how much of the j < i set we'll add to the train set
    div = int(cfg.training_fraction * len(val))
    train += val[:div]
    val = val[div:]
    np.random.shuffle(train)
    return train, val
