"""Base class for models used in the "Lab"

This handles the HookPoint infrastructure and some utility methods that
should be available for all models.
"""
import inspect
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Any, Dict, List, Type

import numpy as np
import torch
import torch.nn as nn

from tlab.utils import NameRepr
from tlab.utils.hookpoint import HookPoint


class LabModel(nn.Module, metaclass=NameRepr):
    @dataclass
    class Config:
        model_class: Type["LabModel"]
        torch_seed: int

        @classmethod
        def from_parent(cls, parent, **kwargs):
            parent_args = {
                k: v
                for k, v in asdict(parent).items()
                if k in inspect.signature(cls).parameters
            }
            parent_args.update(kwargs)
            return cls(**parent_args)

    def __init__(self, cfg: Config):
        super().__init__()
        self.config = cfg
        self.cache = {}

        torch.manual_seed(cfg.torch_seed)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_params})"

    def __getitem__(self, key: str):
        """Get model parameters with convenient, short expression."""
        return self._lookup[key]

    @cached_property
    def _lookup(self) -> Dict[str, torch.Tensor]:
        """Create a cached lookup table for parameters, using short names.
        E.g. model.mlp.layers.0.weight.data -> 'w_l0'
             model.unembed.weight.data -> 'w_u'
             model.block.0.attn.w_q -> ?  TODO
        """
        lookup = {}
        for name, param in self.named_parameters():
            w_type = name.split(".")[-1]
            base = name.split(".")[:-1]
            try:
                mod_name = f"{base[-2][0]}{int(base[-1])}"
            except:
                mod_name = base[-1][0]
            prefix = w_type
            if w_type == "weight":
                prefix = "w"
            elif w_type == "bias":
                prefix = "b"
            key = f"{prefix}_{mod_name}"
            # TODO Do some auto enumeration for name overlap
            assert key not in lookup, f"Param lookup already contains {key}"
            lookup[key] = param.data
        return lookup

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
    def hook_points(self) -> List[nn.Module]:
        return [module for name, module in self.named_modules() if "hook" in name]

    def remove_all_hooks(self) -> None:
        for hp in self.hook_points:
            hp.remove_hooks("fwd")
            hp.remove_hooks("bwd")

    def cache_all(self, cache, include_backward=False) -> None:
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()

        def save_hook_back(tensor, name):
            cache[name + "_grad"] = tensor[0].detach()

        for hp in self.hook_points:
            hp.add_hook(save_hook, "fwd")
            if include_backward:
                hp.add_hook(save_hook_back, "bwd")

    def watch(self, inputs) -> Dict[str, torch.Tensor]:
        cache = {}
        self.cache_all(cache)
        outputs = self(inputs)
        self.remove_all_hooks()
        return outputs, cache
