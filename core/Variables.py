from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class Variable:
    name: str
    kind: str = "uniform"  # 'uniform' | 'normal' | 'lognormal' | 'fixed'
    params: Dict[str, float] = field(default_factory=dict)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.kind == "fixed":
            return np.full(n, self.params.get("value", 0.0), dtype=float)
        if self.kind == "uniform":
            a = self.params.get("low", 0.0)
            b = self.params.get("high", 1.0)
            return rng.uniform(a, b, size=n)
        if self.kind == "normal":
            mu = self.params.get("mean", 0.0)
            sigma = self.params.get("std", 1.0)
            return rng.normal(mu, sigma, size=n)
        if self.kind == "lognormal":
            mean = self.params.get("mean", 0.0)
            sigma = self.params.get("sigma", 1.0)
            return rng.lognormal(mean, sigma, size=n)
        raise ValueError(f"Unknown variable kind: {self.kind}")

@dataclass
class VariableSet:
    variables: List[Variable]

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lows, highs = [], []
        for v in self.variables:
            if v.kind == "uniform":
                lows.append(v.params.get("low", 0.0))
                highs.append(v.params.get("high", 1.0))
            elif v.kind == "fixed":
                val = v.params.get("value", 0.0)
                lows.append(val); highs.append(val)
            else:
                mu = v.params.get("mean", 0.0)
                std = v.params.get("std", v.params.get("sigma", 1.0))
                lows.append(mu - 3*std); highs.append(mu + 3*std)
        return np.array(lows, dtype=float), np.array(highs, dtype=float)

    def names(self) -> List[str]:
        # return a list of variable names
        return [v.name for v in self.variables]

    def _latexify(self) -> List[str]:
        # return a list of latexified variable names
        return [f"${v.name}$" for v in self.variables]

    def to_SAlib(self) -> Dict:
        # return a dictionary of variables in the format of SAlib
        problem = {
            "num_vars": len(self.variables),
            "names": [],
            "bounds": []
        }
        for v in self.variables:
            problem["names"].append(v.name)
            problem["bounds"].append([v.params.get("low", 0.0), v.params.get("high", 1.0)])

        problem["names"] = self._latexify()
        return problem
    
    def to_UQpy(self) -> Dict:
        pass
    

      
        

    
