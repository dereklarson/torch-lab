import numpy as np
import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int, use_bias: bool = True, **kwargs):
        super().__init__()
        self.use_bias = use_bias
        self.weight = nn.Parameter(
            torch.FloatTensor(n_out, n_in).uniform_(-1, 1) / np.sqrt(n_in)
        )
        if self.use_bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(n_out).uniform_(-1, 1) / np.sqrt(n_in)
            )

    def forward(self, x):
        x = x @ self.weight.T
        if self.use_bias:
            x += self.bias
        return x


class Unembed(nn.Module):
    def __init__(self, n_in: int, n_outputs: int):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(n_in, n_outputs) / np.sqrt(n_in))

    def forward(self, x):
        return x @ self.W_U
