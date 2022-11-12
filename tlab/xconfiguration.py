"""Definition of a single experimental setup.

An XConfiguration provides a full picture of every setting for a training run.
It also provides some convenience methods for loading, saving, etc.
An Experiment will specify many XConfigurations.
"""
import pickle
from dataclasses import fields
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Tuple

import parse
import torch
from prettytable import PrettyTable

from tlab.data import DataConfig
from tlab.models.batch_transformer import TransformerConfig
from tlab.models.transformer import Transformer
from tlab.optimize import OptimConfig, Optimizer

VALID_PARAMS = {
    **DataConfig.__annotations__,
    **TransformerConfig.__annotations__,
    **OptimConfig.__annotations__,
}


class XConfiguration:
    def __init__(
        self,
        idx: int,
        data_cfg: DataConfig,
        model_cfg: TransformerConfig,
        optim_cfg: OptimConfig,
        variables: Tuple[str] = tuple(),
    ) -> None:
        self.idx: int = idx
        self.data: DataConfig = data_cfg
        self.model: TransformerConfig = model_cfg
        self.optim: OptimConfig = optim_cfg
        self.variables: Tuple[str] = tuple(sorted(variables))

    @classmethod
    def from_dict(
        cls, idx: int, conf_dict: Dict[str, Any], variables: Tuple[str] = tuple()
    ) -> "XConfiguration":
        """Return an XConfiguration from an index and parameter dictionary"""
        if "seed" not in conf_dict:
            conf_dict["seed"] = idx
        conf_args = []
        for config_class in DataConfig, TransformerConfig, OptimConfig:
            common_param_set = (
                set(f.name for f in fields(config_class)) & conf_dict.keys()
            )
            c_kwargs = {k: conf_dict[k] for k in common_param_set}
            conf_args.append(config_class(**c_kwargs))
        return cls(idx, *conf_args, variables=variables)

    @classmethod
    def load(cls, path: Path) -> "XConfiguration":
        parsed = parse.parse("{root}/{dir}/{idx:d}__{tag}.pkl", str(path))
        with open(path, "rb") as fh:
            params = pickle.load(fh)
        return cls.from_dict(parsed["idx"], params)

    def dump(self, root: Path) -> None:
        with open(root / f"{self.filebase}_cfg.pkl", "wb") as fh:
            pickle.dump(self.params, fh)

    @cached_property
    def params(self) -> Dict[str, Any]:
        """Return raw parameter dictionary from the config dataclasses."""
        params = {}
        params.update(vars(self.data))
        params.update(vars(self.model))
        params.update(vars(self.optim))
        return params

    @property
    def title(self) -> str:
        return ", ".join(f"{p}: {self.params[p]}" for p in self.variables)

    @property
    def codes(self) -> List[Tuple[str, Any]]:
        """Return abbreviated param/value pairs for all varying parameters."""
        codes = []
        for parameter in self.variables:
            par_code = "".join(word[0] for word in parameter.split("_"))
            codes.append((par_code, self.params[parameter]))
        return codes

    @property
    def repr(self) -> str:
        """Convenient descriptor for plots, etc."""
        return ", ".join(f"{code}: {value}" for code, value in self.codes)

    @property
    def filebase(self) -> str:
        tag = "_".join(f"{code}@{value}" for code, value in self.codes)
        if tag == "":
            tag = "Default"
        return f"{self.idx:03}__{tag}"

    def summary(self) -> None:
        """Display pretty output so a user understands the parameters of the run."""
        table = PrettyTable(["Parameter", "Value"])
        for param in sorted(self.params):
            note = "*" if param in self.variables else ""
            table.add_row([f"{note}{param}", self.params[param]])
        print(table)

    def save_model(self, root: Path, model: Transformer, optim: Optimizer):
        filepath = root / f"{self.filebase}_mod.pth"
        save_dict = {
            "params": self.params,
            "model": model.state_dict(),
            "optimizer": optim.optimizer.state_dict(),
            "scheduler": optim.scheduler.state_dict(),
            "train_losses": optim.train_losses,
            "test_losses": optim.test_losses,
            "epoch": optim.epoch,
        }
        torch.save(save_dict, filepath)

    def get_model_state(self, root: Path):
        filepath = root / f"{self.filebase}_mod.pth"
        state_dict = torch.load(filepath)
        return state_dict["model"]

    def load_model(self, root: Path):
        model = Transformer(self.model)
        model.load_state_dict(self.get_model_state(root))
        return model
