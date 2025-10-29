from __future__ import annotations
import math
import numpy as np
from typing import Dict, Any

SAFE_GLOBALS = {
    "__builtins__": {},
    "np": np,
    "math": math,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "pi": math.pi,
    "abs": abs,
    "min": min,
    "max": max,
}

def evaluate_expression(expr: str, local_vars: Dict[str, Any]) -> Any:
    return eval(expr, SAFE_GLOBALS, local_vars)
