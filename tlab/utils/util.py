import functools
import pickle

import numpy as np
import torch


def to_numpy(tensor):
    if type(tensor) == torch.Tensor:
        return tensor.detach().cpu().numpy()
    return tensor


def set_weights(model, param_loc: str, tensor: torch.Tensor, fixed: bool = False):
    param = dict(model.named_parameters())[param_loc]
    param.data = tensor
    if fixed:
        param.requires_grad = False


def set_weights_by_file(model, param_loc: str, tensor_file: str, fixed: bool = False):
    param = dict(model.named_parameters())[param_loc]
    with open(f"weights/{tensor_file}", "rb") as fh:
        param.data = pickle.load(fh)
    if fixed:
        param.requires_grad = False


def cos_k(i, k):
    return torch.cos(2 * torch.pi * torch.arange(k) * i / k)


def normalize(matrix: np.ndarray, axis: int = 1) -> np.ndarray:
    return matrix / matrix.sum(axis=1)[:, np.newaxis]


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


def get_attention_patterns(params):
    W_E = params["model"]["embed.W_E"]
    W_Q = params["model"]["blocks.0.attn.W_Q"]
    W_K = params["model"]["blocks.0.attn.W_K"]
    QK = torch.einsum("ahe,ahE -> aeE", W_Q, W_K)
    eQK = torch.einsum("ve,aeE -> avE", W_E, QK)
    attention = torch.einsum("avE,VE -> avV", eQK, W_E)
    heads = [to_numpy(row).tolist() for row in attention]
    return heads


def get_output_patterns(params):
    W_E = params["model"]["embed.W_E"]
    W_U = params["model"]["unembed.W_U"]
    W_O = params["model"]["blocks.0.attn.W_O"]
    W_V = params["model"]["blocks.0.attn.W_V"]
    OV = torch.einsum("aeh,ahE -> aeE", W_O, W_V)
    uOV = torch.einsum("ev,aeE -> avE", W_U, OV)
    output = torch.einsum("avE,VE -> avV", uOV, W_E)
    heads = [to_numpy(row).tolist() for row in output]
    return heads
