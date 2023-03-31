import functools

import numpy as np
import torch


@functools.lru_cache(maxsize=None)
def fourier_basis(k):
    fourier_basis = []
    fourier_basis.append(torch.ones(k) / np.sqrt(k))
    # Note that if p is even, we need to explicitly add a term for cos(kpi), ie
    # alternating +1 and -1
    for i in range(1, k // 2 + 1):
        fourier_basis.append(torch.cos(2 * torch.pi * torch.arange(k) * i / k))
        fourier_basis.append(torch.sin(2 * torch.pi * torch.arange(k) * i / k))
        fourier_basis[-2] /= fourier_basis[-2].norm()
        fourier_basis[-1] /= fourier_basis[-1].norm()
    return torch.stack(fourier_basis, dim=0).to("cuda")


def tensor2sign(tensor: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Return a tensor containing the sign of each element, if they meet a threshold."""
    tensor = tensor.detach().clone()
    tensor[abs(tensor) <= threshold] = 0
    return torch.sign(tensor)


def sign_similarity(tensor: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Return a tensor containing row-wise sign similarity counts.

    Element A_ij of the result contains the dot product
    """
    signs = tensor2sign(tensor, threshold)
    n_nonzero = signs.abs().sum(1).repeat((signs.shape[0], 1))
    elem_ct = torch.max(n_nonzero, n_nonzero.T)
    prod = (signs @ signs.T) / elem_ct
    return prod.tril(diagonal=-1)
