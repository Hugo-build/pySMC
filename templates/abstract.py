from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from ..core.variables import VariableSet

@dataclass
class AbstractCase:
    name: str
    description: str
    def evaluate(self, x: np.ndarray) -> Dict[str, float]:
        raise NotImplementedError
    @property
    def variables(self) -> VariableSet:
        raise NotImplementedError
