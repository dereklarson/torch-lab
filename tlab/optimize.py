from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from tlab.data import DataDiv, Dataset


@dataclass
class OptimConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    batch: Optional[int] = None
    fixed_wnorm: bool = False
    stopping_threshold: float = -1
    n_epochs: int = 10000


class Optimizer:
    def __init__(self, cfg: OptimConfig, model, loss_func, device="cuda") -> None:
        super().__init__()
        self.config = cfg
        self.model = model
        self.loss_func = loss_func
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.98),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(step / 10, 1)
        )

        self.epoch = 0
        self.start_norm = model.wnorm
        self.train_losses = []
        self.test_losses = []

    def step(self, data: Dataset, device=None) -> None:
        device = device or self.device
        train_loss = self.loss_func(
            self.model, data["Train"]["In"], data["Train"]["Label"], device=device
        )
        test_loss = self.loss_func(
            self.model, data["Test"]["In"], data["Test"]["Label"], device=device
        )
        self.train_losses.append(train_loss.item())
        self.test_losses.append(test_loss.item())

        train_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        if self.config.fixed_wnorm:
            coeff = self.start_norm / self.model.wnorm
            with torch.no_grad():
                for param in self.model.parameters():
                    param *= coeff

        self.epoch += 1

    @property
    def display(self) -> Dict[str, str]:
        return dict(
            lr=f"{self.scheduler.get_last_lr()[0]}",
            train=f"{np.log(self.train_losses[-1]):.4f}",
            test=f"{np.log(self.test_losses[-1]):.4f}",
            gpu=f"{gpu_mem():.3f}",
        )

    def save(self, path: Path) -> None:
        save_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "epoch": self.epoch,
        }
        torch.save(save_dict, path)


def gpu_mem() -> float:
    return torch.cuda.memory_allocated() / 1e9


def cross_entropy(model, inputs: DataDiv, labels: DataDiv, device="cuda"):
    logits = model(inputs)[:, -1]
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    label_tensor = torch.tensor(labels).to(device)
    prediction_logprobs = torch.gather(logprobs, index=label_tensor[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss
