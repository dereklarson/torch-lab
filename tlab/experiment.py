"""The Experiment class provides structure to training a series of DNNs

Exploring the behavior of a system involves probing along its different parameters.
Defining DNNs such as Transformers, training them, and generating synthetic data
each involve a set of parameters that can be tuned. This class and associated codebase
aims to make the experimentation process easier by abstracting away tasks such as
manual configuration, file management, data collection, etc. This saves some time,
but critically also reduces error in the experimental setup.
"""

import itertools
import logging
import operator
import os
import pickle
import shutil
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prettytable import PrettyTable

from tlab.xconfiguration import VALID_PARAMS, XConfiguration


@dataclass
class Relation:
    source: str
    op: str
    value: float


class Experiment:
    root_path: Path = Path("experiments")
    exp_file = "experiment.pkl"
    tag = "temp"

    def __init__(
        self,
        tag: Optional[str] = None,
        defaults: Optional[Dict[str, float]] = None,
    ) -> None:
        self.root_path
        self.tag = tag
        self.defaults = defaults or {}
        for key in self.defaults:
            assert key in VALID_PARAMS.keys(), f"{key} is an invalid parameter"

        self.ranges: Dict[str, Tuple[Any, ...]] = {}
        self.relations: Dict[str, Relation] = {}
        self.observables: Dict[str, Tuple[Any, ...]] = {}

    def __getitem__(self, key: int):
        for cfg in self.configure():
            if cfg.idx == key:
                return cfg

    @property
    def path(self) -> Path:
        return self.root_path / self.tag

    @property
    def count(self) -> int:
        return len(list(self._product()))

    @classmethod
    def _load(cls, path: Path) -> "Experiment":
        with open(path / cls.exp_file, "rb") as fh:
            exp = pickle.load(fh)
        return exp

    @classmethod
    def load(cls, name: str) -> "Experiment":
        """Load by name, using default root path"""
        return cls._load(cls.root_path / name)

    def save(self) -> None:
        with open(self.path / self.exp_file, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def list(
        cls, path: Optional[Path] = None, verbose: bool = False
    ) -> Dict[int, "Experiment"]:
        """Load and summarize all experiments present at 'path'."""
        path = path or cls.root_path
        table = PrettyTable(["Index", "Experiment", "Varied Parameter", "Values"])
        exc_ct = 0
        result = {}
        idx = 0
        for exp_dir in sorted(glob(f"{path}/*")):
            try:
                exp = cls._load(Path(exp_dir))
            except Exception as exc:
                exc_ct += 1
                if verbose:
                    logging.warning(f"Exception loading experiment: {exp_dir}")
                    logging.warning(exc)
                continue
            idx += 1
            result[idx] = exp
            tag = exp.tag
            for param in sorted(exp.ranges.keys()):
                table.add_row([idx, tag, param, exp.ranges[param]])
                tag = ""  # Only print experiment tag once
        print(table)
        print(f"{exc_ct} folders failed to load")
        return result

    def add_relation(self, parameter: str, source: str, op: str, value: float) -> None:
        """Define a parameter that will depend on another via binary op."""
        if parameter not in VALID_PARAMS.keys():
            logging.warning(f"{parameter} is not a defined parameter")
        if source not in VALID_PARAMS.keys():
            logging.warning(f"{source} is not a defined source parameter")
        self.relations[parameter] = Relation(source, op, value)

    def add_range(self, parameter: str, values: tuple) -> None:
        """Define a tuple of values to use for a parameter."""
        if parameter not in VALID_PARAMS.keys():
            logging.warning(f"{parameter} is not a defined parameter")
        self.ranges[parameter] = values

    def summary(self) -> None:
        """Display a pretty summary of the experiment"""
        print(f"Experiment: '{self.tag}':")
        table = PrettyTable(["Parameter", "Value"])
        non_defaults = self.ranges.keys() | self.relations.keys()
        for param in sorted(self.defaults.keys() - non_defaults):
            table.add_row([param, self.defaults[param]])
        for param in sorted(self.relations.keys()):
            rel = self.relations[param]
            table.add_row([f"-> {param}", f"{rel.op}({rel.source}, {rel.value})"])
        for param in sorted(self.ranges.keys()):
            table.add_row([f"*{param}", self.ranges[param]])
        print(table)

    def _product(self) -> List[Dict[str, float]]:
        """Expand all combinations of parameter ranges."""
        if not self.ranges:
            return [{}]

        # Unzip the parameter names and the values they'll take
        keys, value_groups = list(zip(*self.ranges.items()))

        # Fan out the value groups into the individual experimental sets of values
        exp_values = list(itertools.product(*value_groups))

        return (dict(zip(keys, values)) for values in exp_values)

    def _get_params(self, exp_dict: Dict[str, float], **kwargs) -> Dict[str, float]:
        """Get the full parameter dict with defaults overridden by ranged or relational params."""
        full_params = self.defaults.copy()
        full_params.update(exp_dict)
        for param, rel in self.relations.items():
            full_params[param] = getattr(operator, rel.op)(
                full_params[rel.source], rel.value
            )
        return full_params

    def configure(self) -> XConfiguration:
        """Generator for XConfigurations of the Experiment."""
        idx = 0
        for variable_dict in self._product():
            idx += 1
            params = self._get_params(variable_dict)
            variables = tuple(variable_dict.keys())
            yield XConfiguration.from_dict(idx, params, variables)

    def initialize_run(self):
        """Set up file structure for the experiment."""
        if not os.path.isdir(self.path):
            print(f"Creating {self.path}")
            os.mkdir(self.path)
        elif self.tag == Experiment.tag:
            print(f"Recreating temporary directory '{self.tag}'")
            shutil.rmtree(self.path)
            os.mkdir(self.path)
        else:
            logging.warning(f"{self.path} already exists, stopping")
            return

        self.save()
