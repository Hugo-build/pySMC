from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Iterable, List
import numpy as np
import pandas as pd

from .variables import VariableSet
from .samplers import sample_inputs, SamplerKind

@dataclass
class MCResult:
    inputs: pd.DataFrame
    outputs: pd.DataFrame

def run_monte_carlo(
    func: Callable[[np.ndarray], Dict[str, float]] | Callable[[np.ndarray], float],
    vset: VariableSet,
    n: int | None = None,
    sampler: SamplerKind = "sobol",
    seed: Optional[int] = None,
    samples: Optional[Iterable[Iterable[float]]] = None,
) -> MCResult:
    if samples is not None:
        X = np.asarray(list(samples), dtype=float)
        if X.ndim != 2 or X.shape[1] != len(vset.variables):
            raise ValueError("Provided samples must be of shape (N, d) matching VariableSet.")
    else:
        if n is None:
            raise ValueError("n must be provided when samples are not given.")
        X = sample_inputs(vset, n=n, kind=sampler, seed=seed)

    names = vset.names()
    rows: List[Dict[str, float]] = []
    for xi in X:
        yi = func(xi)
        if isinstance(yi, dict):
            rows.append(yi)
        else:
            rows.append({"y": float(yi)})
    inputs_df = pd.DataFrame(X, columns=names)
    outputs_df = pd.DataFrame(rows).reset_index(drop=True)
    return MCResult(inputs=inputs_df, outputs=outputs_df)
