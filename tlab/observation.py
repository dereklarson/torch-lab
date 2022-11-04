import functools
import glob
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import parse
import torch

from tlab.data import DataDiv, Dataset
from tlab.models.transformer import Transformer
from tlab.optimize import Optimizer
from tlab.utils.util import fourier_basis


class Observations:
    def __init__(self) -> None:
        # Define functions for every observable
        self._obs_funcs = {}

    def add_observable(self, obs_name: str, obs_kwargs: Dict[str, Any]) -> None:
        func = getattr(Observables, obs_name)
        if not obs_kwargs:
            self._obs_funcs[obs_name] = func
        elif len(obs_kwargs) == 1:
            arg = list(obs_kwargs)[0]
            if arg in ("device",):
                self._obs_funcs[obs_name] = functools.partial(
                    func, **{arg: obs_kwargs[arg]}
                )
            if arg == "name":
                values = obs_kwargs[arg]
                for val in values:
                    fkey = f"{obs_name}_{val}"
                    self._obs_funcs[fkey] = functools.partial(func, **{arg: val})
        else:
            logging.warning(f"Too many kwargs for {obs_name} observation: {obs_kwargs}")

    def init_run(self) -> None:
        self.data: Dict[str, Any] = defaultdict(list)

    def observe(
        self,
        model: Transformer,
        optim: Optimizer,
        data: Dataset,
        **kwargs,
    ) -> None:
        for key, func in self._obs_funcs.items():
            self.data[key].append(func(model, optim, data, **kwargs))

    def save(self, root: Path, filebase: str):
        obs_path = root / f"{filebase}_obs.pkl"
        with open(obs_path, "wb") as fh:
            pickle.dump(self.data, fh)

    @classmethod
    def load_observations(cls, root: Path, filebase: str):
        obs_path = root / f"{filebase}_obs.pkl"
        try:
            with open(obs_path, "rb") as fh:
                data = pickle.load(fh)
        except FileNotFoundError:
            data = {}
            logging.warning(f"Missing {obs_path}")
        return data

    @classmethod
    def load_obs_group(
        cls, root: Path, verbose: bool = False
    ) -> Dict[int, Dict[str, Any]]:
        data = {}
        for fname in sorted(glob.glob(f"{root}/*_obs.pkl")):
            parsed = parse.parse("{root}/{dir}/{idx:d}__{tag}_obs.pkl", fname)
            # pairs = [tuple(pair.split("@")) for pair in parsed["tag"].split("_")]
            tag = parsed["tag"].replace("_", ", ").replace("@", ": ")
            if verbose:
                print(f"Loading {fname} for {parsed['idx']} | {tag}")
            with open(fname, "rb") as fh:
                data[parsed["idx"]] = pickle.load(fh)
            data[parsed["idx"]]["tag"] = tag
        return data


def _accuracy(model: Transformer, data: DataDiv, device="cuda"):
    inputs = data["In"]
    labels = data["Label"]

    logits = model(inputs)[:, -1]
    _, predictions = torch.max(logits.to(torch.float64), dim=-1)
    label_tensor = torch.tensor(labels).to(device)
    accuracy = (predictions == label_tensor).sum().item() / label_tensor.numel()
    return accuracy


def _fourier_components(
    tensor: torch.Tensor, n_freq: int
) -> Tuple[torch.Tensor, torch.tensor]:
    """Perform a DFT on the tensor and get the strongest frequencies"""
    tensor = tensor.detach()
    weights, indices = (
        (tensor.T @ fourier_basis(n_freq).T).pow(2).sum(0).sort(descending=True)
    )
    return weights, indices


class Observables:
    @staticmethod
    def train_loss(model: Transformer, optim: Optimizer, data, **kwargs) -> float:
        return optim.train_losses[-1]

    @staticmethod
    def test_loss(model: Transformer, optim: Optimizer, data, **kwargs) -> float:
        return optim.test_losses[-1]

    @staticmethod
    def train_accuracy(model: Transformer, optim: Optimizer, data, **kwargs) -> float:
        return _accuracy(model, data["Train"], device=kwargs.get("device", "cuda"))

    @staticmethod
    def test_accuracy(model: Transformer, optim: Optimizer, data, **kwargs) -> float:
        return _accuracy(model, data["Test"], device=kwargs.get("device", "cuda"))

    @staticmethod
    def weight_norm(model: Transformer, optim: Optimizer, data, **kwargs) -> float:
        return float(
            torch.linalg.norm(
                torch.concat([par.flatten() for par in model.parameters()])
            ).to("cpu")
        )

    @staticmethod
    def comp_wnorm(model: Transformer, optim: Optimizer, data, **kwargs) -> float:
        param_name: str = kwargs.get("name", "")
        return float(
            torch.linalg.norm(dict(model.named_parameters())[param_name].flatten()).to(
                "cpu"
            )
        )

    @staticmethod
    def embed_fi_gini(model: Transformer, optim: Optimizer, data, **kwargs) -> float:
        """Calculate the 'Fourier Inverse Gini' coefficient of W_E"""
        fourier_weights, _ = _fourier_components(model.embed.W_E, model.config.n_vocab)
        csum = torch.cumsum(fourier_weights, 0)
        return (len(csum) * csum[-1] / csum.sum()).to("cpu")

    @staticmethod
    def embed_top_components(
        model: Transformer, optim: Optimizer, data, **kwargs
    ) -> float:
        """Calculate the top_k frequencies present in W_E"""
        _, fourier_freqs = _fourier_components(model.embed.W_E, model.config.n_vocab)
        top_k = kwargs.get("top_k", 6)
        return fourier_freqs.to("cpu").numpy()[:top_k]
