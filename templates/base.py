from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Callable
import numpy as np

from ..core.variables import VariableSet

@dataclass
class CaseStudy:
    name: str
    description: str
    variables: VariableSet
    evaluate: Callable[[np.ndarray], Dict[str, float]]

_REGISTRY: Dict[str, CaseStudy] = {}

def register(case: CaseStudy) -> None:
    _REGISTRY[case.name] = case

def get(name: str) -> CaseStudy:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown case: {name}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]

def all_cases() -> List[str]:
    return sorted(_REGISTRY.keys())
