"""ExpOptim adds new methods onto the base LabOptimizer
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from tlab.datasets.lab_dataset import DataBatch
from tlab.models.lab_model import LabModel
from tlab.optimizers.lab_optimizer import LabOptimizer
from tlab.utils.analysis import self_similarity, sign_similarity


class ExpOptim(LabOptimizer):
    @dataclass
    class Config(LabOptimizer.Config):
        shuffle_threshold: float = 0.0  # Perform an update to reduce sign similarity
        repulsion_strength: float = 0.0  # Encourages orthogonality in weight matrices
        repulsion_decay: float = 1.0  # Multiplies repulsive strength every epoch
        experimental_reg: float = 0.0  # Add e.g. L-0.5 or sigmoid regularization

    def __init__(self, cfg: Config, model: LabModel, device="cuda") -> None:
        super().__init__(cfg, model, device)

    def step(self, model: LabModel, batch: DataBatch) -> None:
        super().step(model, batch)

        if self.config.repulsion_strength > 0:
            params = getattr(self, "repulsion_params", [])
            if not params and self.iteration < 5:
                print("Repulsion strength set with no parameters")
            self.repulsion_update(model, params, self.config.repulsion_strength)

        if self.config.experimental_reg > 0:
            coeff = self.config.learning_rate * self.config.experimental_reg
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if not "W_A" in name and not "W_B" in name:
                        continue
                    # L05 reg seems unstable (sign switching for small values)
                    # param -= coeff * torch.sign(param) / param.abs() ** 0.5

                    # Sigmoid reg
                    param -= coeff * (torch.sigmoid(0.1 * param) - 0.5)

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
