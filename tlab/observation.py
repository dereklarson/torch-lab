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
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import parse
import torch

from tlab.datasets.algorithmic import DataBatch, LabDataset
from tlab.models.lab_model import LabModel
from tlab.optimizers.lab_optimizer import LabOptimizer
from tlab.utils.analysis import fourier_basis, self_similarity, sign_similarity


class Observations:
    def __init__(self, observables: Dict[str, bool]) -> None:
        # Define functions for every observable
        self._obs_funcs: Dict[str, Callable] = {}
        self._batch_trigger: Dict[str, bool] = {}

        for observable, always in observables.items():
            self.add_observable(observable, always, {})

        # 'path' and 'data' reset during run initialization
        self.path: Path = Path("./")
        self.data: Dict[str, Any] = defaultdict(dict)

    def add_observable(
        self, obs_name: str, always: bool, obs_kwargs: Dict[str, Any]
    ) -> None:
        func = getattr(Observables, obs_name)
        if not obs_kwargs:
            self._obs_funcs[obs_name] = func
            self._batch_trigger[obs_name] = always
        elif len(obs_kwargs) == 1:
            arg = list(obs_kwargs)[0]
            if arg == "names":
                for val in obs_kwargs[arg]:
                    fkey = f"{obs_name}_{val}"
                    self._obs_funcs[fkey] = functools.partial(func, **{"name": val})
                    self._batch_trigger[fkey] = always
            elif arg == "matrix_elements":
                rows, cols = obs_kwargs[arg]
                for r in range(rows):
                    for c in range(cols):
                        fkey = f"{obs_name}_{r}.{c}"
                        self._obs_funcs[fkey] = functools.partial(func, row=r, col=c)
                        self._batch_trigger[fkey] = always
            else:
                logging.warning(
                    f"Unsupported kwargs for {obs_name} observation: {obs_kwargs}"
                )
        else:
            logging.warning(f"Too many kwargs for {obs_name} observation: {obs_kwargs}")

    def _save(self, data: Any) -> None:
        with open(self.path, "ab") as fh:
            pickle.dump(data, fh)

    def save(self) -> None:
        self._save(self.data)

    def init_run(self, root: Path, filebase: str, header: Dict[str, Any]) -> None:
        self.path = root / f"{filebase}_obs.pkl"
        self.data: Dict[str, Dict[int, float]] = defaultdict(dict)
        self._save(header)

    def observe_once(
        self,
        key,
        func,
        model: LabModel,
        optim: LabOptimizer,
        data: LabDataset,
        **kwargs,
    ) -> None:
        self.data[key][optim.iteration] = func(model, optim, data, **kwargs)

    def observe_batch(
        self,
        model: LabModel,
        optim: LabOptimizer,
        data: LabDataset,
        **kwargs,
    ) -> None:
        for key, func in self._obs_funcs.items():
            if not self._batch_trigger[key]:
                continue
            self.observe_once(key, func, model, optim, data, **kwargs)

    def observe(
        self,
        model: LabModel,
        optim: LabOptimizer,
        data: LabDataset,
        **kwargs,
    ) -> None:
        for key, func in self._obs_funcs.items():
            self.observe_once(key, func, model, optim, data, **kwargs)

    @classmethod
    def load(cls, root: Path, filebase: str, verbose: bool = False):
        """Load one datafile containing observations"""
        obs_path = root / f"{filebase}_obs.pkl"
        data = defaultdict(dict)
        try:
            if verbose:
                print(f"Loading observations {obs_path}...")
            with open(obs_path, "rb") as fh:
                # We assume the first dump will be the header
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


def _accuracy(model: LabModel, batch: DataBatch):
    logits = model(batch.inputs)
    _, predictions = torch.max(logits.to(torch.float64), dim=-1)
    accuracy = (predictions == batch.targets).sum().item() / batch.targets.numel()
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
    def train_loss(model: LabModel, optim: LabOptimizer, data, **kwargs) -> float:
        return optim.train_losses[-1]

    @staticmethod
    def val_loss(
        model: LabModel, optim: LabOptimizer, dataset: LabDataset, **kwargs
    ) -> float:
        losses = []
        for batch in dataset.val_loader:
            losses.append(optim.loss_func(model(batch.inputs), batch.targets).item())
        return np.mean(losses)

    @staticmethod
    def train_accuracy(
        model: LabModel, optim: LabOptimizer, dataset: LabDataset, **kwargs
    ) -> float:
        return _accuracy(model, dataset.get_batch("train"))

    @staticmethod
    def val_accuracy(
        model: LabModel, optim: LabOptimizer, dataset: LabDataset, **kwargs
    ) -> float:
        accuracies = []
        for batch in dataset.val_loader:
            accuracies.append(_accuracy(model, batch))
        return np.mean(accuracies)

    @staticmethod
    def error_rate(
        model: LabModel, optim: LabOptimizer, dataset: tuple, **kwargs
    ) -> float:
        results = []
        for batch in dataset.val_loader:
            results.append(1 - _accuracy(model, batch))
        return np.mean(results)

    @staticmethod
    def sign_similarity(
        model: LabModel, optim: LabOptimizer, data: tuple, **kwargs
    ) -> int:
        tensor = dict(model.named_parameters())[kwargs.get("name")]
        return float(torch.max(sign_similarity(tensor, 0.00)))

    @staticmethod
    def self_similarity(
        model: LabModel, optim: LabOptimizer, data: tuple, **kwargs
    ) -> int:
        tensor = dict(model.named_parameters())[kwargs.get("name")]
        normed_similarity = torch.linalg.norm(self_similarity(tensor))
        normed_similarity /= tensor.shape[0]
        return float(normed_similarity)

    @staticmethod
    def embed_g1(model: LabModel, optim: LabOptimizer, data, **kwargs) -> float:
        # Get the param-wise moving exponential average of the gradient squared
        embed_g2 = optim.optimizer.state_dict()["state"][0]["exp_avg"]
        token_idx: str = kwargs.get("row", 0)
        embed_idx: str = kwargs.get("col", 0)
        return abs(float(embed_g2[token_idx, embed_idx].to("cpu")))

    @staticmethod
    def embed_g2(model: LabModel, optim: LabOptimizer, data, **kwargs) -> float:
        # Get the param-wise moving exponential average of the gradient squared
        embed_g2 = optim.optimizer.state_dict()["state"][0]["exp_avg_sq"]
        token_idx: str = kwargs.get("row", 0)
        embed_idx: str = kwargs.get("col", 0)
        return float(embed_g2[token_idx, embed_idx].to("cpu"))

    @staticmethod
    def weight_norm(model: LabModel, optim: LabOptimizer, data, **kwargs) -> float:
        return float(
            torch.linalg.norm(
                torch.concat([par.flatten() for par in model.parameters()])
            ).to("cpu")
        )

    @staticmethod
    def comp_wnorm(model: LabModel, optim: LabOptimizer, data, **kwargs) -> float:
        param_name: str = kwargs.get("name", "")
        return float(
            torch.linalg.norm(dict(model.named_parameters())[param_name].flatten()).to(
                "cpu"
            )
        )

    @staticmethod
    def embed_hf_fourier(model: LabModel, optim: LabOptimizer, data, **kwargs) -> float:
        """Calculate the highest frequency fourier component of W_E"""
        # There are n_vocab / 2 frequencies, with a cosine and sine term each
        # The constant term is at index 0, the cosine term of highest freq is at n_vocab - 1
        n_comp = model.config.n_vocab
        tensor = model.embed.W_E.detach()
        top_comp, _ = torch.max(fourier_basis(n_comp) @ tensor, axis=1)
        top_comp /= torch.linalg.norm(top_comp)
        return float(top_comp[n_comp - 1])

    @staticmethod
    def embed_fi_gini(model: LabModel, optim: LabOptimizer, data, **kwargs) -> float:
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
        model: LabModel, optim: LabOptimizer, data, **kwargs
    ) -> float:
        """Calculate the top_k frequencies present in W_E"""
        _, fourier_freqs = _fourier_components(
            model.embed.W_E, model.config.n_vocab
        ).sort(descending=True)
        top_k = kwargs.get("top_k", 6)
        return fourier_freqs.to("cpu").numpy()[:top_k]
