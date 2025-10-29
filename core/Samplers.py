from __future__ import annotations
from typing import Literal, Optional
import numpy as np

try:
    from scipy.stats import qmc
except Exception:
    qmc = None

from .Variables import VariableSet

SamplerKind = Literal["random", "lhs", "sobol"]

def sample_inputs(vset: VariableSet, n: int, kind: SamplerKind = "random", seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = len(vset.variables)
    lows, highs = vset.bounds()

    if kind == "random" or qmc is None:
        X_unit = rng.random((n, d))
    elif kind == "lhs":
        eng = qmc.LatinHypercube(d=d, seed=seed)
        X_unit = eng.random(n)
    elif kind == "sobol":
        eng = qmc.Sobol(d=d, scramble=True, seed=seed)
        X_unit = eng.random_base2(int(np.ceil(np.log2(max(n, 2)))))[:n]
    else:
        raise ValueError(f"Unknown sampler kind: {kind}")

    return lows + (highs - lows) * X_unit
