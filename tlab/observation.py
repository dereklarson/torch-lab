"""The Observations class handles extracting and saving model data during training.

As an example, we might be interested in watching the norm of the embedding weights
as a function of training epoch. We can define a function that, given the model, will
calculate this value. We register this function with our Observations instance and
it will keep track of these measurements.

In a small model regime, it can be more useful to keep periodic checkpoints of a model,
assuming this is sufficient granularity. As model size grows, this can require too much
disk space, thus this lean approach. 
"""
import functools
import glob
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import parse
import torch

from tlab.data import DataDiv, Dataset
from tlab.models.lab_model import LabModel
from tlab.optimize import Optimizer
from tlab.utils.analysis import fourier_basis, self_similarity, sign_similarity

STD_OBSERVABLES = [
    "train_loss",
    "test_loss",
    "test_accuracy",
]


class Observations:
    def __init__(self, init=False) -> None:
        # Define functions for every observable
        self._obs_funcs = {}

        # 'data' reset during run initialization
        self.data: Dict[str, Any] = defaultdict(dict)
        self.tag: str = ""

        if init:
            for observable in STD_OBSERVABLES:
                self.add_observable(observable, {})

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
            elif arg == "names":
                for val in obs_kwargs[arg]:
                    fkey = f"{obs_name}_{val}"
                    self._obs_funcs[fkey] = functools.partial(func, **{"name": val})
            elif arg == "matrix_elements":
                rows, cols = obs_kwargs[arg]
                for r in range(rows):
                    for c in range(cols):
                        fkey = f"{obs_name}_{r}.{c}"
                        self._obs_funcs[fkey] = functools.partial(func, row=r, col=c)
            else:
                logging.warning(
                    f"Unsupported kwargs for {obs_name} observation: {obs_kwargs}"
                )
        else:
            logging.warning(f"Too many kwargs for {obs_name} observation: {obs_kwargs}")

    def init_run(self, tag: str) -> None:
        self.tag = tag
        self.data: Dict[str, Dict[int, float]] = defaultdict(dict)

    def observe(
        self,
        model: LabModel,
        optim: Optimizer,
        data: Dataset,
        **kwargs,
    ) -> None:
        for key, func in self._obs_funcs.items():
            self.observe_once(key, func, model, optim, data, **kwargs)

    def observe_once(
        self,
        key,
        func,
        model: LabModel,
        optim: Optimizer,
        data: Dataset,
        **kwargs,
    ) -> None:
        self.data[key][optim.iteration] = func(model, optim, data, **kwargs)

    def save(self, root: Path, filebase: str, header: Dict[str, Any] = None) -> None:
        obs_path = root / f"{filebase}_obs.pkl"
        with open(obs_path, "ab") as fh:
            if header is not None:
                pickle.dump(header, fh)
            pickle.dump(self.data, fh)

    @classmethod
    def load(cls, root: Path, filebase: str, verbose: bool = False):
        """Load one datafile containing observations"""
        obs_path = root / f"{filebase}_obs.pkl"
        data = defaultdict(dict)
        try:
            if verbose:
                print(f"Loading observations {obs_path}...")
            with open(obs_path, "rb") as fh:
                header = pickle.load(fh)
                while True:
                    try:
                        curr = pickle.load(fh)
                        for key in curr:
                            data[key].update(curr[key])
                    except EOFError:
                        break
        except FileNotFoundError:
            data = {}
            logging.warning(f"Missing {obs_path}")

        return data, header

    @classmethod
    def load_obs_group(
        cls, root: Path, verbose: bool = False, filter_strs: Tuple[str, ...] = tuple()
    ) -> Dict[int, Dict[str, Any]]:
        """Load all observations from the same experiment folder. (deprecated)"""
        data = {}
        for fname in sorted(glob.glob(f"{root}/*_obs.pkl")):
            if any([fs not in fname for fs in filter_strs]):
                continue
            parsed = parse.parse("{root}/{dir}/{idx:d}__{tag}_obs.pkl", fname)
            tag = parsed["tag"].replace("_", ", ").replace("@", ": ")
            if verbose:
                print(f"Loading {fname} for {parsed['idx']} | {tag}")
            with open(fname, "rb") as fh:
                data[parsed["idx"]] = pickle.load(fh)
            with open(fname.replace("obs.pkl", "cfg.pkl"), "rb") as fh:
                data[parsed["idx"]]["cfg"] = pickle.load(fh)
            data[parsed["idx"]]["tag"] = tag
        return data


def _accuracy(model: LabModel, inputs, targets):
    logits = model(inputs)
    _, predictions = torch.max(logits.to(torch.float64), dim=-1)
    accuracy = (predictions == targets).sum().item() / targets.numel()
    return accuracy


def _fourier_components(tensor: torch.Tensor, n_freq: int) -> torch.Tensor:
    """Perform a DFT on the tensor, returning the strength of each frequency"""
    tensor = tensor.detach()
    fourier_comp = torch.linalg.norm(fourier_basis(n_freq) @ tensor, axis=1)
    fourier_comp /= torch.linalg.norm(fourier_comp)
    return fourier_comp


class Observables:
    """Collection of functions that can be referenced by add_observable()."""

    @staticmethod
    def train_loss(model: LabModel, optim: Optimizer, data, **kwargs) -> float:
        return optim.train_losses[-1]

    @staticmethod
    def train_loss_batch(model: LabModel, optim: Optimizer, data, **kwargs) -> float:
        return np.mean(optim.train_losses[-10:])

    @staticmethod
    def test_loss(model: LabModel, optim: Optimizer, data, **kwargs) -> float:
        return optim.test_losses[-1]

    @staticmethod
    def test_loss_batch(model: LabModel, optim: Optimizer, data, **kwargs) -> float:
        losses = []
        for inputs, targets in data:
            losses.append(optim.loss_func(model(inputs), targets[:, None]).item())
        return np.mean(losses)

    @staticmethod
    def train_accuracy(
        model: LabModel, optim: Optimizer, data: Dataset, **kwargs
    ) -> float:
        return _accuracy(model, data.train.inputs, data.train.labels)

    @staticmethod
    def test_accuracy_batch(
        model: LabModel, optim: Optimizer, data: tuple, **kwargs
    ) -> float:
        accuracies = []
        for inputs, targets in data:
            accuracies.append(_accuracy(model, inputs, targets))
        return np.mean(accuracies)

    @staticmethod
    def error_rate_batch(
        model: LabModel, optim: Optimizer, data: tuple, **kwargs
    ) -> float:
        results = []
        for inputs, targets in data:
            results.append(1 - _accuracy(model, inputs, targets))
        return np.mean(results)

    @staticmethod
    def sign_similarity(
        model: LabModel, optim: Optimizer, data: tuple, **kwargs
    ) -> int:
        tensor = dict(model.named_parameters())[kwargs.get("name")]
        return float(torch.max(sign_similarity(tensor, 0.00)))

    @staticmethod
    def self_similarity(
        model: LabModel, optim: Optimizer, data: tuple, **kwargs
    ) -> int:
        tensor = dict(model.named_parameters())[kwargs.get("name")]
        normed_similarity = torch.linalg.norm(self_similarity(tensor))
        normed_similarity /= tensor.shape[0]
        return float(normed_similarity)

    @staticmethod
    def test_accuracy(
        model: LabModel, optim: Optimizer, data: Dataset, **kwargs
    ) -> float:
        return _accuracy(model, data.test.inputs, data.test.labels)

    @staticmethod
    def embed_g1(model: LabModel, optim: Optimizer, data, **kwargs) -> float:
        # Get the param-wise moving exponential average of the gradient squared
        embed_g2 = optim.optimizer.state_dict()["state"][0]["exp_avg"]
        token_idx: str = kwargs.get("row", 0)
        embed_idx: str = kwargs.get("col", 0)
        return abs(float(embed_g2[token_idx, embed_idx].to("cpu")))

    @staticmethod
    def embed_g2(model: LabModel, optim: Optimizer, data, **kwargs) -> float:
        # Get the param-wise moving exponential average of the gradient squared
        embed_g2 = optim.optimizer.state_dict()["state"][0]["exp_avg_sq"]
        token_idx: str = kwargs.get("row", 0)
        embed_idx: str = kwargs.get("col", 0)
        return float(embed_g2[token_idx, embed_idx].to("cpu"))

    @staticmethod
    def weight_norm(model: LabModel, optim: Optimizer, data, **kwargs) -> float:
        return float(
            torch.linalg.norm(
                torch.concat([par.flatten() for par in model.parameters()])
            ).to("cpu")
        )

    @staticmethod
    def comp_wnorm(model: LabModel, optim: Optimizer, data, **kwargs) -> float:
        param_name: str = kwargs.get("name", "")
        return float(
            torch.linalg.norm(dict(model.named_parameters())[param_name].flatten()).to(
                "cpu"
            )
        )

    @staticmethod
    def embed_hf_fourier(model: LabModel, optim: Optimizer, data, **kwargs) -> float:
        """Calculate the highest frequency fourier component of W_E"""
        # There are n_vocab / 2 frequencies, with a cosine and sine term each
        # The constant term is at index 0, the cosine term of highest freq is at n_vocab - 1
        n_comp = model.config.n_vocab
        tensor = model.embed.W_E.detach()
        top_comp, _ = torch.max(fourier_basis(n_comp) @ tensor, axis=1)
        top_comp /= torch.linalg.norm(top_comp)
        return float(top_comp[n_comp - 1])

    @staticmethod
    def embed_fi_gini(model: LabModel, optim: Optimizer, data, **kwargs) -> float:
        """Calculate the 'Fourier Inverse Gini' coefficient of W_E"""
        n_comp = model.config.n_vocab
        tensor = model.embed.W_E.detach()
        fourier_weights = torch.linalg.norm(fourier_basis(n_comp) @ tensor, axis=1)
        fourier_weights /= torch.linalg.norm(fourier_weights)
        fourier_weights.sort(descending=True)
        csum = torch.cumsum(fourier_weights, 0)
        return (len(csum) * csum[-1] / csum.sum()).to("cpu")

    @staticmethod
    def embed_top_components(
        model: LabModel, optim: Optimizer, data, **kwargs
    ) -> float:
        """Calculate the top_k frequencies present in W_E"""
        _, fourier_freqs = _fourier_components(
            model.embed.W_E, model.config.n_vocab
        ).sort(descending=True)
        top_k = kwargs.get("top_k", 6)
        return fourier_freqs.to("cpu").numpy()[:top_k]
