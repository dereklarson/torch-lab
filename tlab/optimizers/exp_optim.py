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

    def __init__(self, cfg: Config, model: LabModel, device="cuda") -> None:
        super().__init__(cfg, model, device)

    def step(self, model: LabModel, batch: DataBatch) -> None:
        super().step(model, batch)

        if self.config.repulsion_strength > 0:
            params = getattr(self, "repulsion_params", [])
            if not params and self.iteration < 5:
                print("Repulsion strength set with no parameters")
            self.repulsion_update(model, params, self.config.repulsion_strength)

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
