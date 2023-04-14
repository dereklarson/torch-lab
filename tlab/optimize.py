"""Optimizer contains the configuration and functionality for operating the training.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from tlab.data import Dataset
from tlab.models.lab_model import LabModel
from tlab.utils.analysis import self_similarity, sign_similarity


@dataclass
class OptimConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    n_epochs: int = 10000
    manual_decay: float = 0.0
    adam_betas: tuple = (0.90, 0.98)
    fixed_wnorm: bool = False  # Rescale weights after each update so norm is fixed
    shuffle_threshold: float = 0.0  # Perform an update to reduce sign similarity
    repulsion_strength: float = 0.0  # Encourages orthogonality in weight matrices
    repulsion_decay: float = 1.0  # Multiplies repulsive strength every epoch


class Optimizer:
    def __init__(
        self, cfg: OptimConfig, model: LabModel, loss_func, device="cuda"
    ) -> None:
        super().__init__()
        self.config = cfg
        self.loss_func = loss_func
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=cfg.adam_betas,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(step / 10, 1)
        )

        self.epoch = 0
        self.iteration = 0
        self.train_losses = []
        self.test_losses = []

    def measure_loss(self, model: LabModel, data: Dataset, device=None):
        device = device or self.device
        train_logits = model(data.train.inputs)
        train_loss = self.loss_func(
            train_logits, data.train.labels[:, None], device=device
        )
        test_logits = model(data.test.inputs)
        test_loss = self.loss_func(
            test_logits, data.test.labels[:, None], device=device
        )
        self.train_losses.append(train_loss.item())
        self.test_losses.append(test_loss.item())
        return train_loss, test_loss

    def epoch_step(self, model: LabModel, data: Dataset, device=None) -> None:
        """Process one training step: handle loss, learning_rate, etc
        Use for full batch training.
        TODO: Find a clean way to fold into batched training
        """
        self.optimizer.zero_grad()
        train_loss, _ = self.measure_loss(model=model, data=data, device=device)
        train_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.config.manual_decay:
            with torch.no_grad():
                for param in model.parameters():
                    param -= (
                        self.config.learning_rate * self.config.manual_decay * param
                    )

        if self.config.repulsion_strength > 0:
            params = getattr(self, "repulsion_params", [])
            if not params and self.epoch == 0:
                print("Repulsion strength set with no parameters")
            self.repulsion_update(model, params, self.config.repulsion_strength)

        self.epoch += 1
        self.iteration += 1

    def repulsion_update(
        self, model: LabModel, params: List[str], strength: float = 0.001
    ):
        for param in params:
            weights = dict(model.named_parameters())[param]
            with torch.no_grad():
                ss = self_similarity(weights)
                diag_mag = torch.sum(ss, dim=0)
                delta = diag_mag[:, None] * weights - torch.einsum(
                    "ki,kj->ij", ss, weights
                )
                weights += strength * delta

    def sign_shuffle(
        self,
        model: LabModel,
        params: List[str],
        epsilon: float = 0.0,
    ) -> None:
        if self.config.shuffle_threshold == 0.0:
            return
        with torch.no_grad():
            for param in params:
                tensor = dict(model.named_parameters())[param]
                # Produce a mask for the rows that are too similar to other earlier rows
                sim_rows = (
                    sign_similarity(tensor.detach(), epsilon).max(1)[0]
                    > self.config.shuffle_threshold
                )
                new_rows = torch.randn((sum(sim_rows), tensor.shape[1])) / np.sqrt(
                    tensor.shape[1]
                )
                tensor[sim_rows] = new_rows.to("cuda")

    def step(self, model: LabModel, inputs, labels) -> None:
        self.optimizer.zero_grad()
        train_logits = model(inputs)
        train_loss = self.loss_func(train_logits, labels)
        self.train_losses.append(train_loss.item())
        train_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.iteration += 1

    def end_epoch(self, model: LabModel, test_loader) -> None:
        """Process one training step: handle loss, learning_rate, etc"""
        model.eval()
        batch_losses = []
        for inputs, labels in test_loader:
            batch_losses.append(self.loss_func(model(inputs), labels[:, None]).item())

        self.test_losses.append(np.mean(batch_losses))
        self.epoch += 1

    @property
    def display(self) -> Dict[str, str]:
        """Postfix for TQDM progress bar, to track key variables"""
        display_entries = dict(
            gpu=f"{gpu_mem():.3f}",
            lr=f"{self.scheduler.get_last_lr()[0]}",
            train=f"{np.log(self.train_losses[-1]):.4f}",
        )
        if len(self.test_losses) > 0:
            display_entries["test"] = f"{np.log(self.test_losses[-1]):.4f}"
        return display_entries

    @property
    def epoch_display(self) -> Dict[str, str]:
        """Postfix for TQDM progress bar, to track key variables"""
        display_entries = dict(
            gpu=f"{gpu_mem():.3f}",
            train=f"{np.log(self.train_losses[-1]):.4f}",
        )
        if len(self.test_losses) > 0:
            display_entries["test"] = f"{np.log(self.test_losses[-1]):.4f}"
        return display_entries


def gpu_mem() -> float:
    return torch.cuda.memory_allocated() / 1e9


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor, device="cuda"):
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels, dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss
