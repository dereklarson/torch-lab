"""LabOptimizer contains the configuration and functionality for operating the training.
"""
import math
from dataclasses import dataclass
from typing import Callable, Type

import torch
import torch.nn.functional as F

from tlab.datasets.lab_dataset import DataBatch
from tlab.models.lab_model import LabModel
from tlab.utils import NameRepr


class LabOptimizer(metaclass=NameRepr):
    @dataclass
    class Config:
        optim_class: Type["LabOptimizer"]
        loss_func: Callable = F.cross_entropy
        n_epochs: int = 10000
        learning_rate: float = 1e-3
        warmup_iters: int = 10
        decay_iters: int = 0
        final_coeff: float = 1.0
        weight_decay: float = 1.0
        adam_betas: tuple = (0.90, 0.98)
        torch_dtype: str = "float32"
        grad_clip: float = 0.0

    def __init__(self, cfg: Config, model: LabModel, device="cuda") -> None:
        self.config = cfg
        self.loss_func = cfg.loss_func
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=cfg.adam_betas,
        )

        def annealing(step):
            if step < cfg.warmup_iters:
                return step / cfg.warmup_iters
            if step > cfg.decay_iters:
                return cfg.final_coeff
            decay_ratio = (step - cfg.warmup_iters) / (
                cfg.decay_iters - cfg.warmup_iters
            )
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            return cfg.final_coeff + coeff * (1 - cfg.final_coeff)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, annealing)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.torch_dtype == "float16"))

        self.epoch = 0
        self.iteration = 0
        self.train_losses = []
        self.observed_loss = None

    def measure_loss(self, model: LabModel, batch: DataBatch) -> torch.Tensor:
        train_loss = self.loss_func(model(batch.inputs), batch.targets)
        self.train_losses.append(train_loss.item())
        self.observed_loss = self.train_losses[-1]
        return train_loss

    def step(self, model: LabModel, batch: DataBatch) -> None:
        self.optimizer.zero_grad()
        train_loss = self.measure_loss(model, batch)

        self.scaler.scale(train_loss).backward()
        if self.config.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        self.iteration += 1

    def end_epoch(self, model: LabModel) -> None:
        """Process one training step: handle loss, learning_rate, etc"""
        model.eval()
        self.epoch += 1
