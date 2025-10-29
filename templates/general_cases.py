from __future__ import annotations
from typing import Dict, Any, List, Optional, Sequence
from dataclasses import dataclass
import numpy as np
from ..core.variables import VariableSet
from ..core.safe_eval import evaluate_expression
from .abstract import AbstractCase

# Algebraic
@dataclass
class AlgebraicCase(AbstractCase):
    _variables: VariableSet
    expression: str
    @property
    def variables(self) -> VariableSet:
        return self._variables
    def evaluate(self, x: np.ndarray) -> Dict[str, float]:
        loc = {name: float(val) for name, val in zip(self.variables.names(), x)}
        y = evaluate_expression(self.expression, loc)
        if isinstance(y, dict):
            return {k: float(v) for k, v in y.items()}
        return {"y": float(y)}

# ODE
try:
    from scipy.integrate import solve_ivp
except Exception:
    solve_ivp = None

@dataclass
class ODECase(AbstractCase):
    _variables: VariableSet
    state_dim: int
    rhs: List[str]
    t_span: Sequence[float]
    t_eval: Optional[Sequence[float]] = None
    outputs: Optional[Dict[str, str]] = None
    @property
    def variables(self) -> VariableSet:
        return self._variables
    def evaluate(self, x: np.ndarray) -> Dict[str, float]:
        if solve_ivp is None:
            raise RuntimeError("scipy is required for ODECase but not available.")
        params = {name: float(val) for name, val in zip(self.variables.names(), x)}
        def f(t, y):
            loc = {"t": float(t), "y": np.asarray(y, dtype=float), **params}
            return [evaluate_expression(expr, loc) for expr in self.rhs]
        sol = solve_ivp(f, (float(self.t_span[0]), float(self.t_span[1])),
                        y0=np.zeros(self.state_dim), t_eval=self.t_eval, dense_output=self.t_eval is None)
        if self.t_eval is not None and sol.y.size > 0:
            t = np.asarray(self.t_eval, dtype=float)
            X = sol.y.T
        else:
            t = np.linspace(self.t_span[0], self.t_span[1], 201)
            X = sol.sol(t).T  # type: ignore
        if not self.outputs:
            return {f"x{i}_final": float(X[-1, i]) for i in range(X.shape[1])}
        loc = {"t": t, "X": X, **params, "np": np}
        return {k: float(evaluate_expression(expr, loc)) for k, expr in self.outputs.items()}

# External
import json, subprocess, tempfile, os
@dataclass
class ExternalCase(AbstractCase):
    _variables: VariableSet
    command: List[str]
    input_format: str = "json"
    output_format: str = "json"
    output_key_map: Optional[Dict[str, str]] = None
    @property
    def variables(self) -> VariableSet:
        return self._variables
    def evaluate(self, x: np.ndarray) -> Dict[str, float]:
        params = {name: float(val) for name, val in zip(self.variables.names(), x)}
        with tempfile.TemporaryDirectory() as td:
            inp_path = os.path.join(td, "input.json")
            out_path = os.path.join(td, "output.json")
            with open(inp_path, "w", encoding="utf-8") as f:
                json.dump({"params": params}, f)
            cmd = [c.format(json_path=inp_path, out_path=out_path, **params) for c in self.command]
            subprocess.run(cmd, check=True)
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        if self.output_key_map:
            return {new: float(data[old]) for old, new in self.output_key_map.items()}
        return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
