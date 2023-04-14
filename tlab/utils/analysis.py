import collections
import functools
from typing import Set

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


def random_with_exclusion(minimum: int, maximum: int, exclusion: Set[int]) -> int:
    """Return a random value, minimum <= val < maximum, that's not in exclusion"""
    if maximum < 100000:
        options = set(range(minimum, maximum)) - exclusion
        return np.random.choice(list(options))
    else:
        # Untenable to create our option list for large maximum values, so we should
        # rely on sparsity and just retry until we get a valid number.
        choice = np.random.randint(minimum, maximum)
        ct = 0
        while choice in exclusion and ct < 100:
            choice = np.random.randint(minimum, maximum)
            ct += 1
        assert ct < 100, "Too many tries for random_with_exclusion"
        return choice


def sign_index(tensor: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Return a tensor containing the sign of each element, if they meet a threshold."""
    pow2 = torch.tensor([2**i for i in range(tensor.shape[1])])
    sign_mask = (torch.sign(tensor) + 1) / 2
    indices = (sign_mask * pow2).sum(axis=1)
    return indices.int()


def index2sign(idx: int, N: int) -> torch.Tensor:
    """Return a vector of length N and sign index 'idx'"""
    base = -np.ones(N)
    for idx, digit in enumerate(bin(idx)[2:][::-1]):
        if digit == "1":
            base[idx] = 1
    return torch.tensor(base)


def distinguish_signs(tensor: torch.Tensor) -> torch.Tensor:
    M, N = tensor.shape
    # Numpy cast required as torch.Tensor elements seem to be considered unique even if
    # their value is equivalent.
    indices = sign_index(tensor).numpy()
    used = set(indices)
    to_fix = {key: False for key, ct in collections.Counter(indices).items() if ct > 1}
    for sign_idx, row_idx in zip(indices, range(M)):
        if sign_idx not in to_fix:
            continue
        if not to_fix[sign_idx]:
            to_fix[sign_idx] = True
            continue
        new_idx = random_with_exclusion(0, 2**N, used)
        used.add(new_idx)
        tensor[row_idx] = tensor[row_idx].abs() * index2sign(new_idx, N)
    return tensor


def sign_similarity(tensor: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Return a tensor containing row-wise sign similarity counts."""
    signs = tensor2sign(tensor, threshold)
    n_nonzero = signs.abs().sum(1).repeat((signs.shape[0], 1))
    elem_ct = torch.max(n_nonzero, n_nonzero.T)
    prod = (signs @ signs.T) / elem_ct
    return prod.tril(diagonal=-1)


def self_similarity(tensor: torch.Tensor) -> torch.Tensor:
    """Return a tensor containing row-wise similarity magnitudes.

    Element A_ij of the result contains the dot product of rows i and j.
    This is clamped to non-negative values, i.e. vectors at opposing
    directions yield 0 similarity.
    """
    normed = (tensor.T / torch.linalg.norm(tensor, dim=1)).T
    positive_prod = torch.clamp(normed @ normed.T, min=0, max=None)
    return positive_prod.fill_diagonal_(0)
