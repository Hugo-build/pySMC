from __future__ import annotations
import json, pathlib
from typing import Dict, Any, List, Optional, Sequence
from ..core.variables import Variable, VariableSet
from ..templates.general_cases import AlgebraicCase, ODECase, ExternalCase
from ..templates.abstract import AbstractCase

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

def _vset_from_spec(spec: List[Dict[str, Any]]) -> VariableSet:
    return VariableSet([
        Variable(name=v["name"], kind=v.get("kind","uniform"), params=v.get("params",{}))
        for v in spec
    ])

def load_case_from_file(path: str | pathlib.Path) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if p.suffix.lower() in (".yaml",".yml"):
        if yaml is None:
            raise RuntimeError("pyyaml not installed; cannot read YAML.")
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    return json.loads(p.read_text(encoding="utf-8"))

def build_case_from_config(cfg: Dict[str, Any]) -> AbstractCase:
    name = cfg.get("name","case")
    desc = cfg.get("description","")
    vset = _vset_from_spec(cfg["variables"])
    model = cfg["model"]
    mtype = model["type"]

    if mtype == "algebraic":
        return AlgebraicCase(name=name, description=desc, _variables=vset,
                             expression=model["expression"])

    if mtype == "ode":
        return ODECase(name=name, description=desc, _variables=vset,
                       state_dim=model["state_dim"],
                       rhs=model["rhs"],
                       t_span=model.get("t_span",[0.0,1.0]),
                       t_eval=model.get("t_eval", None),
                       outputs=model.get("outputs", None))

    if mtype == "external":
        return ExternalCase(name=name, description=desc, _variables=vset,
                            command=model["command"],
                            output_key_map=model.get("output_key_map", None))

    raise ValueError(f"Unknown model.type: {mtype}")

def get_sampling_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "samples" in cfg and cfg["samples"] is not None:
        return {"samples": cfg["samples"]}
    mc = cfg.get("monte_carlo", {})
    return {"n": mc.get("n",256), "sampler": mc.get("sampler","sobol"), "seed": mc.get("seed", None)}
