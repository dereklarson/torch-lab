"""Optimizer contains the configuration and functionality for operating the training.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from tlab.data import Dataset
from tlab.models.transformer import Transformer


@dataclass
class OptimConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    n_epochs: int = 10000
    fixed_wnorm: bool = False  # Rescale weights after each update so norm is fixed
    manual_decay: float = 0.0
    adam_betas: tuple = (0.90, 0.98)


class Optimizer:
    def __init__(
        self, cfg: OptimConfig, model: Transformer, loss_func, device="cuda"
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
        self.train_losses = []
        self.test_losses = []

    def measure_loss(self, model: Transformer, data: Dataset, device=None):
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

    def step(self, model: Transformer, data: Dataset, device=None) -> None:
        """Process one training step: handle loss, learning_rate, etc"""
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

        self.epoch += 1

    @property
    def display(self) -> Dict[str, str]:
        """Postfix for TQDM progress bar, to track key variables"""
        return dict(
            lr=f"{self.scheduler.get_last_lr()[0]}",
            train=f"{np.log(self.train_losses[-1]):.4f}",
            test=f"{np.log(self.test_losses[-1]):.4f}",
            gpu=f"{gpu_mem():.3f}",
        )


def gpu_mem() -> float:
    return torch.cuda.memory_allocated() / 1e9


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor, device="cuda"):
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels, dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss
