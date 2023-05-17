"""The Experiment class provides structure to training a series of DNNs

Exploring the behavior of a system involves probing along its different parameters.
Defining DNNs such as Transformers, training them, and generating synthetic data
each involve a set of parameters that can be tuned. This class and associated codebase
aims to make the experimentation process easier by abstracting away tasks such as
manual configuration, file management, data collection, etc. This saves some time,
but critically also reduces error in the experimental setup.
"""

import itertools
import json
import logging
import operator
import os
import pickle
import shutil
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import pandas as pd
from prettytable import PrettyTable

from tlab.datasets import LabDataset
from tlab.models.lab_model import LabModel
from tlab.observation import Observations
from tlab.optimizers.lab_optimizer import LabOptimizer
from tlab.utils.util import StopExecution
from tlab.xconfiguration import XConfiguration


@dataclass
class Relation:
    source: str
    op: Callable
    arg: float


class Experiment:
    root_path: Path = Path("experiments")
    exp_file = "experiment.pkl"
    temp_tag = "temp"

    def __init__(
        self,
        tag: Optional[str] = None,
        defaults: Optional[Dict[str, float]] = None,
    ) -> None:
        self.tag = tag
        self.defaults = defaults or {}

        assert "dataset_class" in defaults, "'defaults' must specify 'dataset_class'"
        self.dataset_class: Type[LabDataset] = defaults["dataset_class"]
        assert "model_class" in defaults, "'defaults' must specify 'model_class'"
        self.model_class: Type[LabModel] = defaults["model_class"]
        assert "optim_class" in defaults, "'defaults' must specify 'optim_class'"
        self.optim_class: Type[LabOptimizer] = defaults["optim_class"]

        self.valid_params = XConfiguration.valid_params(
            self.dataset_class, self.model_class, self.optim_class
        )
        for key in self.defaults:
            assert key in self.valid_params.keys(), f"{key} is an invalid parameter"

        self.ranges: Dict[str, Tuple[Any, ...]] = {}
        self.relations: Dict[str, Relation] = {}

    def __getitem__(self, key: int):
        for cfg in self.configure():
            if cfg.idx == key:
                return cfg

    @property
    def count(self) -> int:
        return len(list(self._product()))

    @property
    def path(self) -> Path:
        return self.root_path / self.tag

    @classmethod
    def _load(cls, path: Path) -> "Experiment":
        with open(path / cls.exp_file, "rb") as fh:
            exp = pickle.load(fh)
        return exp

    @classmethod
    def load(cls, name: str) -> "Experiment":
        """Load by name, using default root path"""
        return cls._load(cls.root_path / name)

    def change_root(self, dir_name: str) -> None:
        self.root_path = Path(dir_name)

    def save(self) -> None:
        with open(self.path / self.exp_file, "wb") as fh:
            pickle.dump(self, fh)

    def rename(self, new_tag: str) -> "Experiment":
        """Rename and move files associated with the Experiment."""
        new_path = self.root_path / new_tag
        os.rename(self.path, new_path)
        self.tag = new_tag
        self.save()
        return self

    @classmethod
    def _glob_exp(cls, path: Optional[Path] = None, verbose: bool = False):
        exc_ct = 0
        path = path or cls.root_path
        for exp_dir in sorted(glob(f"{path}/*")):
            try:
                exp = cls._load(Path(exp_dir))
                yield exp
            except Exception as exc:
                exc_ct += 1
                if verbose:
                    logging.warning(f"Exception loading experiment: {exp_dir}")
                    logging.warning(exc)
                continue

    @classmethod
    def list(
        cls, path: Optional[Path] = None, verbose: bool = False
    ) -> Dict[int, "Experiment"]:
        """Load and summarize all experiments present at 'path'."""
        table = PrettyTable(["Index", "Experiment", "Varied Parameter", "Values"])
        result = {}
        for idx, exp in enumerate(cls._glob_exp(path=path, verbose=verbose)):
            result[idx] = exp
            tag = exp.tag
            for param in sorted(exp.ranges.keys()):
                values = exp.ranges[param][:10]
                if len(exp.ranges[param]) > 10:
                    values += ("...",)
                table.add_row([idx, tag, param, values])
                tag = ""  # Only print experiment tag once
            if not exp.ranges:
                table.add_row([idx, tag, "None", "Default"])

        print(table)
        # print(f"{exc_ct} folders failed to load")
        return result

    @classmethod
    def match(cls, substr: str, path: Optional[Path] = None) -> "Experiment":
        for exp in cls._glob_exp(path=path, verbose=False):
            if substr in exp.tag:
                return exp
        logging.warning(f"No experiment found with tag containing {substr}")

    def add_relation(self, parameter: str, source: str, op: str, value: float) -> None:
        """Define a parameter that will depend on another via binary op."""
        if parameter not in self.valid_params.keys():
            logging.warning(f"{parameter} is not a defined parameter")
        if source not in self.valid_params.keys():
            logging.warning(f"{source} is not a defined source parameter")
        self.relations[parameter] = Relation(source, op, value)

    def add_range(self, parameter: str, values: tuple) -> None:
        """Define a tuple of values to use for a parameter."""
        if parameter not in self.valid_params.keys():
            logging.warning(f"{parameter} is not a defined parameter")
        self.ranges[parameter] = values

    def summary(self) -> None:
        """Display a pretty summary of the experiment"""
        print(f"Experiment: '{self.tag}' with {self.count} configurations:")
        table = PrettyTable(["Parameter", "Value"])
        non_defaults = self.ranges.keys() | self.relations.keys()
        for param in sorted(self.defaults.keys() - non_defaults):
            table.add_row([param, self.defaults[param]])
        for param in sorted(self.relations.keys()):
            rel = self.relations[param]
            table.add_row(
                [f"-> {param}", f"{rel.op.__name__}({rel.source}, {rel.arg})"]
            )
        for param in sorted(self.ranges.keys()):
            values = self.ranges[param][:10]
            if len(self.ranges[param]) > 10:
                values += ("...",)
            table.add_row([f"*{param}", values])
        print(table)

    def initialize(self, force: bool = False):
        """Set up file structure for the experiment."""
        if not os.path.isdir(self.path):
            print(f"Creating {self.path}")
            os.mkdir(self.path)
        elif force or self.tag == Experiment.temp_tag:
            print(f"Recreating directory '{self.tag}'")
            shutil.rmtree(self.path)
            os.mkdir(self.path)
        else:
            logging.warning(f"{self.path} already exists, stopping")
            raise StopExecution

        self.save()

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
            if not callable(rel.op):
                raise NotImplementedError(f"Can't handle relation with op {rel.op}")
            src = full_params[rel.source]
            full_params[param] = rel.op(src, rel.arg)

        return full_params

    def configure(self) -> XConfiguration:
        """Generator for XConfigurations of the Experiment."""
        idx = 0
        for variable_dict in self._product():
            idx += 1
            params = self._get_params(variable_dict)
            variables = tuple(variable_dict.keys())
            yield XConfiguration.from_dict(
                idx,
                dataset_class=self.dataset_class,
                model_class=self.model_class,
                optim_class=self.optim_class,
                conf_dict=params,
                variables=variables,
            )

    def prepare_runs(self, obs: Observations) -> XConfiguration:
        """Generate Xconfigurations and initialize the run as well."""
        for xcon in self.configure():
            xcon.dump(self.path)
            obs.init_run(self.path, xcon.filebase, header=xcon.params)
            yield xcon

    def load_state(self, idx: int, epoch: Optional[int] = None):
        return self[idx].get_model_state(self.path, epoch)

    def load_observations(
        self, verbose: bool = False, filter_strs: Tuple[str, ...] = tuple()
    ) -> List[pd.DataFrame]:
        """Load all observations from the same experiment."""
        series = []
        for xcon in self.configure():
            if any([fs not in xcon.tag for fs in filter_strs]):
                continue
            data, cfg = Observations.load(self.path, xcon.filebase, verbose=verbose)
            df = pd.DataFrame(data).assign(
                **{k: cfg[k] for k in cfg & self.ranges.keys()}
            )
            df["exp_idx"] = xcon.idx
            series.append(df)

        return series

    def load_final_results(
        self, verbose: bool = False, filter_strs: Tuple[str, ...] = tuple()
    ) -> pd.DataFrame:
        """Load only the final values for measurements from an experiment."""
        series_list = self.load_observations(verbose=verbose, filter_strs=filter_strs)
        df = pd.DataFrame([_df.iloc[-1] for _df in series_list]).astype(
            {"exp_idx": "int32"}
        )
        return df.set_index("exp_idx")

    def dump_json_data(self, dest: Path, name: str = "Default", **kwargs):
        # Dump a list of configurations
        with open(dest / f"{name}.json", "w") as fh:
            dump_params = {
                "name": name.title(),
                "tags": [xcon.tag for xcon in self.configure()],
                "notes": kwargs.get("desc", ""),
            }
            json.dump(dump_params, fh)

        # Dump the configuration and frame data for each xcon
        for xcon in self.configure():
            xcon.dump_json_data(self.path, dest, name)
