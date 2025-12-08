from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import numpy as np
import copy
import re

@dataclass
class Variable:
    name: str
    kind: str = "uniform"  # 'uniform' | 'normal' | 'lognormal' | 'fixed'
    params: Dict[str, float] = field(default_factory=dict)
    targets: List[Dict[str, Any]] = field(default_factory=list)  # Config injection targets
    default: Optional[float] = None  # Default value if not sampled
    description: Optional[str] = None  # Description of the variable
    unit: Optional[str] = None  # Unit of the variable

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
    
    def add_target(self, doc: str, path: str) -> None:
        """Add a target location for config injection.
        
        Args:
            doc: Document/config name (e.g., 'physics', 'FE_config')
            path: Path string (e.g., 'elements[0].E', 'loads[29]')
        """
        self.targets.append({"doc": doc, "path": path})
    
    def has_targets(self) -> bool:
        """Check if variable has any targets defined."""
        return len(self.targets) > 0

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
    
    def inject_values(
        self, 
        configs: Dict[str, Dict[str, Any]], 
        values: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Inject variable values into configuration dictionaries.
        
        Args:
            configs: Dictionary of config dicts, keyed by document name
                    e.g., {"FE_config": {...}, "solver": {...}}
            values: Dictionary mapping variable names to values
                    e.g., {"E": 205e3, "load": 1000}
        
        Returns:
            New configs dict with injected values (deep copy)
        """
        new_configs = copy.deepcopy(configs)
        
        for var in self.variables:
            if var.name not in values:
                continue
            
            value = values[var.name]
            
            for target in var.targets:
                doc_name = target["doc"]
                path = target["path"]
                
                if doc_name not in new_configs:
                    raise ValueError(f"Document '{doc_name}' not found in configs")
                
                _inject_at_path(new_configs[doc_name], path, value)
        
        return new_configs
    
    def sample_configs(
        self,
        configs: Dict[str, Dict[str, Any]],
        n_samples: int,
        rng: np.random.Generator
    ) -> Tuple[List[Dict[str, Dict[str, Any]]], np.ndarray]:
        """
        Generate n sampled configurations.
        
        Args:
            configs: Base configuration dictionaries
            n_samples: Number of samples to generate
            rng: Random number generator
        
        Returns:
            (sampled_configs, samples) where:
                - sampled_configs: List of config dicts
                - samples: Array of shape (n_samples, n_vars)
        """
        samples = np.zeros((n_samples, len(self.variables)))
        for i, var in enumerate(self.variables):
            samples[:, i] = var.sample(n_samples, rng)
        
        sampled_configs = []
        for sample in samples:
            values = {var.name: sample[i] for i, var in enumerate(self.variables)}
            sampled_configs.append(self.inject_values(configs, values))
        
        return sampled_configs, samples


# ============================================================================
# Path Injection Utilities
# ============================================================================

def _parse_path(path: str) -> List[Tuple[str, Union[int, str, None]]]:
    """
    Parse path string into list of (key, index) tuples.
    
    Supports:
        - "key" -> [(key, None)]
        - "key[0]" -> [(key, 0)]
        - "key[*]" -> [(key, "*")]
        - "key[0].attr" -> [(key, 0), (attr, None)]
        - "elements[*].E" -> [("elements", "*"), ("E", None)]
    
    Examples:
        >>> _parse_path("elements[0].E")
        [("elements", 0), ("E", None)]
        >>> _parse_path("loads[29]")
        [("loads", 29)]
        >>> _parse_path("elements[*].E")
        [("elements", "*"), ("E", None)]
    """
    tokens = []
    # Pattern: word followed by optional [number or *] followed by optional .
    pattern = r'(\w+)(?:\[(\d+|\*)\])?\.?'
    
    for match in re.finditer(pattern, path):
        key = match.group(1)
        index = match.group(2)
        
        if index is not None:
            if index == "*":
                tokens.append((key, "*"))
            else:
                tokens.append((key, int(index)))
        else:
            tokens.append((key, None))
    
    return tokens


def _inject_at_path(config: Dict, path: str, value: Any) -> None:
    """
    Inject value at specified path in config dictionary.
    Modifies config in place.
    
    Args:
        config: Configuration dictionary
        path: Path string (e.g., "elements[0].E", "loads[29]")
        value: Value to inject
    """
    tokens = _parse_path(path)
    _set_nested(config, tokens, value)


def _set_nested(obj: Any, tokens: List[Tuple], value: Any) -> None:
    """
    Set value at nested location specified by tokens.
    Modifies obj in place.
    """
    if not tokens:
        return
    
    key, index = tokens[0]
    
    if len(tokens) == 1:
        # Last token - set the value
        if index == "*":
            # Apply to all elements
            for item in obj[key]:
                if isinstance(item, dict):
                    # For list of dicts, this is setting a key at the wrong level
                    # We need the key from the next level
                    # This case should not happen if path is well-formed
                    raise ValueError("Wildcard path must have nested attribute")
                else:
                    # For list of objects with attributes
                    setattr(item, key, value)
        elif index is not None:
            # Indexed access: obj[key][index] = value
            obj[key][index] = value
        else:
            # Direct key access: obj[key] = value
            obj[key] = value
    else:
        # Navigate deeper
        if index == "*":
            # Apply to all elements recursively
            for item in obj[key]:
                _set_nested(item, tokens[1:], value)
        elif index is not None:
            # Navigate through indexed element
            _set_nested(obj[key][index], tokens[1:], value)
        else:
            # Navigate through key
            _set_nested(obj[key], tokens[1:], value)


def inject_single_config(
    config: Dict[str, Any],
    variables: List[Variable],
    values: Dict[str, float]
) -> Dict[str, Any]:
    """
    Helper function to inject values into a single config.
    Useful when you have only one config document.
    
    Args:
        config: Configuration dictionary
        variables: List of Variable objects with targets
        values: Dictionary mapping variable names to values
    
    Returns:
        New config with injected values (deep copy)
    """
    new_config = copy.deepcopy(config)
    
    for var in variables:
        if var.name not in values:
            continue
        
        value = values[var.name]
        
        for target in var.targets:
            path = target["path"]
            _inject_at_path(new_config, path, value)
    
    return new_config

