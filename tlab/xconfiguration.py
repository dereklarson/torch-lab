"""Definition of a single experimental setup.

An XConfiguration provides a full picture of every setting for a training run.
It also provides some convenience methods for loading, saving, etc.
An Experiment will specify many XConfigurations.
"""
import glob
import json
import pickle
from dataclasses import fields
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, get_type_hints

import numpy as np
import parse
import torch
from prettytable import PrettyTable

from tlab.data import DataConfig, Dataset
from tlab.models.lab_model import LabModel, ModelConfig
from tlab.observation import Observations
from tlab.optimize import OptimConfig, Optimizer
from tlab.utils.util import (
    get_attention_patterns,
    get_mlp,
    get_output_patterns,
    get_ov,
    get_qk,
    to_numpy,
)


class XConfiguration:
    def __init__(
        self,
        idx: int,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        optim_cfg: OptimConfig,
        variables: Tuple[str] = tuple(),
    ) -> None:
        self.idx: int = idx

        self.init_seeds(data_cfg, model_cfg)

        # TODO Allow a set of DataConfigs to specify composite datasets
        self.data: DataConfig = data_cfg
        self.model: ModelConfig = model_cfg
        self.optim: OptimConfig = optim_cfg
        self.variables: Tuple[str] = tuple(sorted(variables))

    def init_seeds(self, data_cfg: DataConfig, model_cfg: ModelConfig) -> None:
        """Generate new random seeds for Numpy and PyTorch if not specified."""
        if data_cfg.data_seed == 0:
            rng = np.random.default_rng()
            data_cfg.data_seed = rng.integers(1, 0xFFFFFFFFFFFF)
        if model_cfg.torch_seed == 0:
            rng = np.random.default_rng()
            model_cfg.torch_seed = rng.integers(1, 0xFFFFFFFFFFFF)

    @staticmethod
    def valid_params(model_cfg: ModelConfig):
        return {
            **get_type_hints(DataConfig),
            **get_type_hints(model_cfg),
            **get_type_hints(OptimConfig),
        }

    @classmethod
    def from_dict(
        cls,
        idx: int,
        model_config_class: Type[ModelConfig],
        conf_dict: Dict[str, Any],
        variables: Tuple[str] = tuple(),
    ) -> "XConfiguration":
        """Return an XConfiguration from an index and parameter dictionary"""
        conf_args = []
        for config_class in (
            DataConfig,
            model_config_class,
            OptimConfig,
        ):
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
    def tag(self) -> str:
        tag = "_".join(f"{code}@{value}" for code, value in self.codes)
        if tag == "":
            tag = "default"
        return tag

    @property
    def filebase(self) -> str:
        safe_tag = self.tag.replace("[", "").replace("]", "")
        return f"{self.idx:03}__{safe_tag}"

    @property
    def repr(self) -> str:
        """Convenient descriptor for plots, etc."""
        return ", ".join(f"{code}: {value}" for code, value in self.codes)

    def summary(self) -> None:
        """Display pretty output so a user understands the parameters of the run."""
        table = PrettyTable(["Parameter", "Value"])
        for param in sorted(self.params):
            note = "*" if param in self.variables else ""
            table.add_row([f"{note}{param}", self.params[param]])
        print(table)

    def checkpoint_model(
        self,
        root: Path,
        model: LabModel,
        optim: Optimizer,
        observations: Observations,
    ):
        filepath = root / f"{self.filebase}_mdl_{optim.epoch:0>6d}.pth"
        save_dict = {
            "params": self.params,
            "model": model.state_dict(),
            "train_loss": optim.train_losses[-1],
            "test_loss": optim.test_losses[-1],
            "test_accuracy": observations.data.get("test_accuracy", [0])[-1],
            "epoch": optim.epoch,
        }
        torch.save(save_dict, filepath)

    def save_model(self, root: Path, model: LabModel, optim: Optimizer):
        filepath = root / f"{self.filebase}_mdl.pth"
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

    def get_model_state(self, root: Path, epoch: Optional[int] = None):
        if epoch is not None:
            filepath = root / f"{self.filebase}_mdl_{epoch:0>6d}.pth"
        else:
            filepath = root / f"{self.filebase}_mdl.pth"
        state_dict = torch.load(filepath)
        return state_dict["model"]

    def load_model_checkpoints(self, root: Path, period: int = 100):
        for checkpoint_file in sorted(glob.glob(f"{root}/{self.filebase}_mdl_*.pth")):
            state_dict = torch.load(checkpoint_file)
            yield state_dict

    def load_model(self, root: Path, model_class: Type[LabModel]):
        model = model_class(self.model)
        model.load_state_dict(self.get_model_state(root))
        return model

    def dump_json_data(
        self, src: Path, dest: Path, dataset: Dataset, name: str = "Default", **kwargs
    ):
        # First dump the configuration
        with open(dest / f"{name}__{self.tag}__config.json", "w") as fh:
            dump_params = {
                "name": name,
                "vocabulary": dataset.vocabulary,
            }
            dump_params.update(self.params)
            json.dump(dump_params, fh)

        # Then dump the data for each checkpoint as a frame
        frames = []
        for params in self.load_model_checkpoints(src):
            has_mlp = self.model.d_mlp > 0
            frames.append(
                {
                    "epoch": params["epoch"],
                    "lossTrain": params["train_loss"],
                    "lossTest": params["test_loss"],
                    "accuracyTest": params["test_accuracy"],
                    "embedding": to_numpy(params["model"]["embed.W_E"]).tolist(),
                    "pos_embed": to_numpy(
                        params["model"]["position_embed.W_pos"]
                    ).tolist(),
                    "unembedding": to_numpy(params["model"]["unembed.W_U"]).tolist(),
                    "blocks": [
                        _get_block_params(params, i, has_mlp)
                        for i in range(self.model.n_blocks)
                    ],
                }
            )
        with open(dest / f"{name}__{self.tag}__frames.json", "w") as fh:
            json.dump(frames, fh)


def _get_block_params(params, index: int, has_mlp: bool) -> Dict[str, Any]:
    return {
        "qk": get_qk(params, index),
        "ov": get_ov(params, index),
        "attention": get_attention_patterns(params, index),
        "output": get_output_patterns(params, index),
        "mlp": [] if not has_mlp else get_mlp(params, index),
    }
