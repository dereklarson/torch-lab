"""Optimizer contains the configuration and functionality for operating the training.
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tlab.datasets.dataset import DataBatch, Dataset
from tlab.models.lab_model import LabModel
from tlab.utils.analysis import self_similarity, sign_similarity


@dataclass
class OptimConfig:
    n_epochs: int = 10000
    learning_rate: float = 1e-3
    warmup_iters: int = 10
    decay_iters: int = 5000
    final_coeff: float = 0.1
    weight_decay: float = 1.0
    adam_betas: tuple = (0.90, 0.98)
    torch_dtype: str = "float32"
    grad_clip: float = 0.0
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

    def measure_loss(self, model: LabModel, batch: DataBatch) -> torch.Tensor:
        train_loss = self.loss_func(model(batch.inputs), batch.targets)
        self.train_losses.append(train_loss.item())
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

        if self.config.repulsion_strength > 0:
            params = getattr(self, "repulsion_params", [])
            if not params and self.epoch == 0:
                print("Repulsion strength set with no parameters")
            self.repulsion_update(model, params, self.config.repulsion_strength)

        self.iteration += 1

    def end_epoch(self, model: LabModel) -> None:
        """Process one training step: handle loss, learning_rate, etc"""
        model.eval()
        self.epoch += 1

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
