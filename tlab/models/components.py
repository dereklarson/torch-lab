import numpy as np
import torch
import torch.nn as nn

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
        self.W_E = nn.Parameter(torch.randn(n_vocab, d_embed) / np.sqrt(d_embed))

    def forward(self, x):
        return self.W_E[x, :]


class Unembed(nn.Module):
    def __init__(self, n_in: int, n_outputs: int):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(n_outputs, n_in) / np.sqrt(n_in))

    def forward(self, x):
        return x @ self.W_U.T


class PositionEmbed(nn.Module):
    def __init__(self, n_ctx: int, d_embed: int):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(n_ctx, d_embed) / np.sqrt(d_embed))

    def forward(self, x):
        return x + self.W_pos
