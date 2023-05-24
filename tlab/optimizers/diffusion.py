"""
An implementation of a Denoising Diffusion Probabilistic Model, based on:
https://github.com/TeaPearce/Conditional_Diffusion_MNIST

This is essentially a specialized set of optimization operations that
train a UNet to predict the noise in an image, with included conditioning
from an embedding of the target class.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from tlab.datasets.lab_dataset import DataBatch
from tlab.models.lab_model import LabModel
from tlab.optimizers.lab_optimizer import LabOptimizer


class DDPM(LabOptimizer):
    @dataclass
    class Config(LabOptimizer.Config):
        ddpm_betas: Tuple[float, float] = (1e-4, 0.02)
        n_timesteps: int = 400
        p_dropout: float = 0.1

    def __init__(self, cfg: Config, model: LabModel, device: str = "cuda"):
        """Pre-compute schedules for the DDPM sampling and training process."""
        super().__init__(cfg, model, device)
        beta1, beta2 = cfg.ddpm_betas
        n_T = cfg.n_timesteps
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

        beta_t = (beta2 - beta1) * torch.arange(
            0, n_T + 1, dtype=torch.float32
        ) / n_T + beta1
        beta_t = beta_t.to(device)

        self.sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        self.alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        self.sqrtab = torch.sqrt(self.alphabar_t)
        self.oneover_sqrta = 1 / torch.sqrt(alpha_t)

        self.sqrtmab = torch.sqrt(1 - self.alphabar_t)
        self.mab_over_sqrtmab = (1 - alpha_t) / self.sqrtmab

    def step(self, model: LabModel, batch: DataBatch) -> None:
        self.optimizer.zero_grad()
        x, c = batch.inputs, batch.targets
        _ts = torch.randint(1, self.config.n_timesteps + 1, (x.shape[0],)).to(
            self.device
        )  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c) + self.config.p_dropout).to(
            self.device
        )

        noise_estimate = model(x_t, c, _ts / self.config.n_timesteps, context_mask)
        train_loss = self.loss_func(noise_estimate, noise)
        train_loss.backward()

        # Use an exponential moving average for the observed loss
        if self.observed_loss is None:
            self.observed_loss = train_loss.item()
        else:
            self.observed_loss = 0.95 * self.observed_loss + 0.05 * train_loss.item()

        self.optimizer.step()
        self.scheduler.step()
        self.iteration += 1

    def sample(self, model: LabModel, n_sample: int, size, guide_w=0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(
            self.device
        )  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0, 10).to(
            self.device
        )  # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample / c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(self.device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.0  # makes second half of batch context free

        x_i_store = []  # keep track of generated steps in case want to plot something
        for i in range(self.config.n_timesteps, 0, -1):
            t_is = torch.tensor([i / self.config.n_timesteps]).to(self.device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            # double batch
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            eps = model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i % 20 == 0 or i == self.config.n_timesteps or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

    def evaluate(
        self,
        model: LabModel,
        w: int,
        n_sample: int,
    ):
        model.eval()
        with torch.no_grad():
            x_gen, x_gen_store = self.sample(
                model, n_sample * model.config.n_classes, (1, 28, 28), guide_w=w
            )

        return x_gen, x_gen_store
