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
        data: DataConfig,
        model: TransformerConfig,
        optim: OptimConfig,
        variables: Tuple[str] = tuple(),
    ) -> None:
        self.idx: int = idx
        self.data: DataConfig = data
        self.model: TransformerConfig = model
        self.optim: OptimConfig = optim
        self.variables: Tuple[str] = tuple(sorted(variables))

    @classmethod
    def from_dict(
        cls, idx: int, conf_dict: Dict[str, Any], variables: Tuple[str] = tuple()
    ) -> "XConfiguration":
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
        params = {}
        params.update(vars(self.data))
        params.update(vars(self.model))
        params.update(vars(self.optim))
        return params

    @property
    def codes(self) -> List[Tuple[str, Any]]:
        codes = []
        for parameter in self.variables:
            par_code = "".join(word[0] for word in parameter.split("_"))
            codes.append((par_code, self.params[parameter]))
        return codes

    @property
    def repr(self) -> str:
        return ", ".join(f"{code}: {value}" for code, value in self.codes)

    @property
    def filebase(self) -> str:
        tag = "_".join(f"{code}@{value}" for code, value in self.codes)
        if tag == "":
            tag = "Default"
        return f"{self.idx:03}__{tag}"

    def summary(self) -> None:
        table = PrettyTable(["Parameter", "Value"])
        for param in sorted(self.params):
            note = "*" if param in self.variables else ""
            table.add_row([f"{note}{param}", self.params[param]])
        print(table)

    def save(self, root: Path, model: Transformer, optim: Optimizer):
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
