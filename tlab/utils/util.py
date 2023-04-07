import copy
import pickle
from typing import Optional

import numpy as np
import torch

from tlab.models.lab_model import LabModel


class StopExecution(Exception):
    def _render_traceback_(self):
        pass


def gen_sign_combinations(n_col: int) -> np.ndarray:
    """Return an array with 2^n rows of all sign combinations of n values"""
    return np.array(np.meshgrid(*([[-1, 1]] * n_col))).T.reshape(-1, n_col)


def to_numpy(tensor):
    if type(tensor) in (torch.Tensor, torch.nn.parameter.Parameter):
        return tensor.detach().cpu().numpy()
    return tensor


def to_torch(matrix):
    if type(matrix) not in (torch.Tensor, torch.nn.parameter.Parameter):
        matrix = torch.tensor(matrix)
    return matrix.to("cuda")


def set_weights(model: LabModel, param_loc: str, data, fixed: bool = False):
    param = dict(model.named_parameters())[param_loc]
    param.data = to_torch(data)
    if fixed:
        param.requires_grad = False


def set_weights_by_file(
    model: LabModel, param_loc: str, tensor_file: str, fixed: bool = False
):
    param = dict(model.named_parameters())[param_loc]
    with open(f"weights/{tensor_file}", "rb") as fh:
        param.data = pickle.load(fh)
    if fixed:
        param.requires_grad = False


def ablate_model(
    model: LabModel, param: str, row: Optional[int] = None, col: Optional[int] = None
):
    """Zero out a row or column from a specified parameter in a model copy"""
    ablated_model = copy.deepcopy(model)
    with torch.no_grad():
        weights = dict(ablated_model.named_parameters())[param]
        if row is not None:
            weights.data[row] = 0
        elif col is not None:
            weights.data[:, col] = 0
    return ablated_model


def add_row(tensor: torch.Tensor, val: float) -> torch.Tensor:
    return torch.nn.ConstantPad1d((0, 0, 0, 1), val)(tensor)


def add_col(tensor: torch.Tensor, val: float) -> torch.Tensor:
    return torch.nn.ConstantPad1d((0, 1, 0, 0), val)(tensor)


def cos_k(i, k):
    return torch.cos(2 * torch.pi * torch.arange(k) * i / k)


def normalize(matrix: np.ndarray, axis: int = 1) -> np.ndarray:
    return matrix / matrix.sum(axis=1)[:, np.newaxis]


def get_qk(params, block_index: int):
    W_Q = params["model"][f"blocks.{block_index}.attn.W_Q"]
    W_K = params["model"][f"blocks.{block_index}.attn.W_K"]
    QK = torch.einsum("ahe,ahE -> aeE", W_Q, W_K)
    return to_numpy(QK).tolist()


def get_ov(params, block_index: int):
    W_O = params["model"][f"blocks.{block_index}.attn.W_O"]
    W_V = params["model"][f"blocks.{block_index}.attn.W_V"]
    OV = torch.einsum("aeh,ahE -> aeE", W_O, W_V)
    return to_numpy(OV).tolist()


def get_attention_patterns(params, block_index: int):
    W_E = params["model"]["embed.W_E"]
    W_Q = params["model"][f"blocks.{block_index}.attn.W_Q"]
    W_K = params["model"][f"blocks.{block_index}.attn.W_K"]
    QK = torch.einsum("ahe,ahE -> aeE", W_Q, W_K)
    eQK = torch.einsum("ve,aeE -> avE", W_E, QK)
    attention = torch.einsum("avE,VE -> avV", eQK, W_E)
    heads = [to_numpy(row).tolist() for row in attention]
    return heads


def get_output_patterns(params, block_index: int):
    W_E = params["model"]["embed.W_E"]
    W_U = params["model"]["unembed.W_U"]
    W_O = params["model"][f"blocks.{block_index}.attn.W_O"]
    W_V = params["model"][f"blocks.{block_index}.attn.W_V"]
    OV = torch.einsum("aeh,ahE -> aeE", W_O, W_V)
    uOV = torch.einsum("ev,aeE -> avE", W_U, OV)
    output = torch.einsum("avE,VE -> avV", uOV, W_E)
    heads = [to_numpy(row).tolist() for row in output]
    return heads


def get_mlp(params, block_index: int):
    return [
        {
            "weights": to_numpy(
                params["model"][f"blocks.{block_index}.mlp.W_in"]
            ).tolist(),
            "biases": to_numpy(
                params["model"][f"blocks.{block_index}.mlp.b_in"]
            ).tolist(),
        },
        {
            "weights": to_numpy(
                params["model"][f"blocks.{block_index}.mlp.W_out"]
            ).tolist(),
            "biases": to_numpy(
                params["model"][f"blocks.{block_index}.mlp.b_out"]
            ).tolist(),
        },
    ]
