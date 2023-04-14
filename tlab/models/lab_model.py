"""Base class for models used in the "Lab"

This handles the HookPoint infrastructure and some utility methods that
should be available for all models.
"""
import inspect
from dataclasses import asdict, dataclass
from typing import List, Type

import numpy as np
import torch
import torch.nn as nn

from tlab.utils.hookpoint import HookPoint


@dataclass
class ModelConfig:
    torch_seed: int


class LabModel(nn.Module):
    @dataclass
    class Config:
        model_class: Type["LabModel"]
        torch_seed: int

        @classmethod
        def from_parent(cls, parent):
            return cls(
                **{
                    k: v
                    for k, v in asdict(parent).items()
                    if k in inspect.signature(cls).parameters
                }
            )

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.cache = {}

        torch.manual_seed(cfg.torch_seed)

    def _init_hooks(self):
        # Call in child class at the end of init
        for name, module in self.named_modules():
            if type(module) == HookPoint:
                module.give_name(name)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def shape(self) -> List:
        return [(n, list(p.shape)) for n, p in self.named_parameters()]

    @property
    def wnorm(self) -> float:
        return float(
            torch.linalg.norm(
                torch.concat([par.flatten() for par in self.parameters()])
            ).to("cpu")
        )

    @property
    def hook_points(self):
        return [module for name, module in self.named_modules() if "hook" in name]

    def remove_all_hooks(self):
        for hp in self.hook_points:
            hp.remove_hooks("fwd")
            hp.remove_hooks("bwd")

    def cache_all(self, cache, include_backward=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()

        def save_hook_back(tensor, name):
            cache[name + "_grad"] = tensor[0].detach()

        for hp in self.hook_points:
            hp.add_hook(save_hook, "fwd")
            if include_backward:
                hp.add_hook(save_hook_back, "bwd")
