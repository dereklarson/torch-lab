"""Optimizer contains the configuration and functionality for operating the training.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tlab.datasets.dataset import DataBatch, Dataset
from tlab.models.lab_model import LabModel
from tlab.utils.analysis import self_similarity, sign_similarity
from tlab.utils.util import gpu_mem


@dataclass
class OptimConfig:
    n_epochs: int = 10000
    learning_rate: float = 1e-3
    warmup_iters: int = 10
    final_lr: float = 0.1
    weight_decay: float = 1.0
    adam_betas: tuple = (0.90, 0.98)
    torch_dtype: str = "float16"
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
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(step / cfg.warmup_iters, 1)
        )

        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.torch_dtype == "float16"))

        self.epoch = 0
        self.iteration = 0
        self.train_losses = []
        self.curr_val_loss = 4

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

    def end_epoch(self, model: LabModel, dataset: Dataset) -> None:
        """Process one training step: handle loss, learning_rate, etc"""
        model.eval()
        batch_losses = []
        for batch in dataset.val_loader:
            batch_losses.append(
                self.loss_func(model(batch.inputs), batch.targets).item()
            )

        self.epoch += 1
        self.curr_val_loss = np.mean(batch_losses)

    def display(self, entries: Tuple[str, ...] = tuple()) -> Dict[str, str]:
        """Postfix for TQDM progress bar, to track key optimization variables."""
        display_entries = dict(
            train=f"{np.log(self.train_losses[-1]):.4f}",
            eval=f"{np.log(self.curr_val_loss):.4f}",
        )
        if "lr" in entries:
            display_entries["lr"] = f"{self.scheduler.get_last_lr()[0]}"
        if "gpu" in entries:
            display_entries["gpu"] = f"{gpu_mem():.3f}"
        return display_entries

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
