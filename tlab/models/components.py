import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tlab.utils.analysis import distinguish_signs


class LinearLayer(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        init_method: str = "kaiming",
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.use_bias = use_bias
        init_tensor = torch.empty(n_out, n_in)

        if init_method == "orthogonal":
            nn.init.orthogonal_(init_tensor)
        elif init_method == "sign_kai":
            nn.init.kaiming_uniform_(init_tensor, nonlinearity="relu")
            distinguish_signs(init_tensor)
        else:
            nn.init.kaiming_uniform_(init_tensor, nonlinearity="relu")

        self.weight = nn.Parameter(init_tensor)

        if self.use_bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(n_out).uniform_(-1, 1) / np.sqrt(n_in)
            )

    def forward(self, x):
        x = x @ self.weight.T
        if self.use_bias:
            x += self.bias
        return x


class Embed(nn.Module):
    def __init__(self, n_vocab: int, d_embed: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_vocab, d_embed) / np.sqrt(d_embed))

    def forward(self, x):
        return self.weight[x, :]


class Unembed(nn.Module):
    def __init__(self, n_in: int, n_outputs: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_outputs, n_in) / np.sqrt(n_in))

    def forward(self, x):
        return x @ self.weight.T


class PositionEmbed(nn.Module):
    def __init__(self, n_ctx: int, d_embed: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_ctx, d_embed) / np.sqrt(d_embed))

    def forward(self, x):
        return x + self.weight


class LayerNorm(nn.Module):
    """As torch.nn.LayerNorm but with an optional bias."""

    def __init__(self, n_dim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))
        self.bias = nn.Parameter(torch.zeros(n_dim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps=1e-5)
