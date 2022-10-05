from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from tlab.data import DataDiv, Dataset


@dataclass
class OptConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1.0
    batch: Optional[int] = None
    stopping_threshold: float = -1
    n_epochs: int = 10000


class Optimizer:
    def __init__(self, cfg: OptConfig, model, loss_func) -> None:
        super().__init__()
        self.config = cfg
        self.model = model
        self.loss_func = loss_func

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
        self.train_losses = []
        self.test_losses = []

    def step(self, data: Dataset) -> None:
        train_loss = self.loss_func(
            self.model, data["Train"]["In"], data["Train"]["Label"]
        )
        test_loss = self.loss_func(self.model, data["Val"]["In"], data["Val"]["Label"])
        self.train_losses.append(train_loss.item())
        self.test_losses.append(test_loss.item())

        train_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        self.epoch += 1

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


def cross_entropy(model, inputs: DataDiv, labels: DataDiv):
    logits = model(inputs)[:, -1]
    label_tensor = torch.tensor(labels).to("cuda")
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=label_tensor[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss


def plot_loss(train_losses, test_losses):
    plt.plot(np.log(train_losses), label="T")
    plt.plot(np.log(test_losses), label="V")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.legend()
