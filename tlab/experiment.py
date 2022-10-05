import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional

from tlab.data import DataConfig
from tlab.models.transformer import TransformerConfig
from tlab.optimize import OptConfig

VALID_PARAMS = {
    **DataConfig.__annotations__,
    **TransformerConfig.__annotations__,
    **OptConfig.__annotations__,
}


class Experiment:
    def __init__(self, exp_dict: Optional[Dict[str, tuple]] = None) -> None:
        self.exp_dict = exp_dict or {}

    def add_range(self, parameter: str, values: tuple) -> None:
        assert parameter in VALID_PARAMS.keys()
        self.exp_dict[parameter] = values

    def product(self) -> List[Dict[str, float]]:
        # Unzip the parameter names and the values they'll take
        keys, value_groups = list(zip(*self.exp_dict.items()))

        # Fan out the value groups into the individual experimental sets of values
        exp_values = list(itertools.product(*value_groups))

        return (dict(zip(keys, values)) for values in exp_values)
