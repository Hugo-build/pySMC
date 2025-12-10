"""
Manifest-based configuration loading and value injection.

Provides utilities to:
1. Load multiple config files specified in a manifest
2. Inject sampled values into configs according to a VariableSet
3. Generate modified configs for batch simulations
"""

from __future__ import annotations
import json
import copy
from pathlib import Path
from typing import Any, Dict, List, Union

from ..core.Variables import Variable, VariableSet, inject_single_config


def load_configs_from_manifest(manifest_path: Path | str) -> Dict[str, Any]:
    """
    Load all config files specified in a manifest as raw dicts.
    
    The manifest JSON should have the structure:
        {
            "workspace_path": "optional/relative/path",
            "files": {
                "env": "env_config.json",
                "lineSys": "system.json",
                ...
            }
        }
    
    Files are loaded from the same directory as the manifest.
    
    Args:
        manifest_path: Path to the INPUT_manifest.json file
        
    Returns:
        Dict with keys matching manifest file keys, values are loaded JSON dicts.
        Missing files have None as value.
        
    Example:
        >>> configs = load_configs_from_manifest("usr0/INPUT_manifest.json")
        >>> configs["env"]["wave"]["Hs"]
        5.0
    """
    manifest_path = Path(manifest_path)
    base_dir = manifest_path.parent
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    configs = {}
    
    for key, filename in manifest.get("files", {}).items():
        file_path = base_dir / filename
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                configs[key] = json.load(f)
        else:
            configs[key] = None
    
    return configs


def inject_from_varset(
    configs: Dict[str, Any],
    var_set: VariableSet,
    values: Dict[str, float],
) -> Dict[str, Any]:
    """
    Inject values into configs according to VariableSet targets.
    
    Each variable in var_set has targets specifying paths like:
        "env.wave.Hs" -> configs["env"]["wave"]["Hs"]
        "env.current.vel[0]" -> configs["env"]["current"]["vel"][0]
    
    Args:
        configs: Dict of config dicts loaded from manifest
        var_set: VariableSet with variables defining injection targets
        values: Dict mapping variable names to values to inject
        
    Returns:
        New configs dict with injected values (deep copy)
    """
    return inject_single_config(configs, var_set.variables, values)


def inject_cases(
    configs: Dict[str, Any],
    var_set: VariableSet,
    cases: List[List[float]],
    var_names: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Inject multiple cases into configs, returning list of modified configs.
    
    Args:
        configs: Dict of config dicts loaded from manifest
        var_set: VariableSet with variables defining injection targets
        cases: List of cases, each case is a list of values
        var_names: Optional list of variable names corresponding to case columns.
                   If None, uses names from var_set.variables in order.
                   
    Returns:
        List of modified config dicts, one per case
        
    Example:
        >>> cases = [[5.0, 11.0], [3.0, 9.0]]  # [Hs, Tp] values
        >>> results = inject_cases(configs, var_set, cases, ["Hs", "Tp"])
        >>> len(results)
        2
    """
    if var_names is None:
        var_names = [v.name for v in var_set.variables]
    
    results = []
    for case in cases:
        values = dict(zip(var_names, case))
        injected = inject_from_varset(configs, var_set, values)
        results.append(injected)
    
    return results


def save_injected_config(
    config: Dict[str, Any],
    output_path: Path | str,
    key: str = None,
) -> None:
    """
    Save an injected config (or sub-config) to a JSON file.
    
    Args:
        config: Config dict to save
        output_path: Path to output JSON file
        key: Optional key to save only a sub-config (e.g., "env")
    """
    output_path = Path(output_path)
    data = config[key] if key else config
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class ManifestLoader:
    """
    Convenience class for working with manifest-based configs.
    
    Example:
        >>> loader = ManifestLoader("INPUT_manifest.json", var_set)
        >>> loader.inject({"Hs": 5.0, "Tp": 11.0})
        >>> loader.configs["env"]["wave"]["Hs"]
        5.0
    """
    
    def __init__(
        self,
        manifest_path: Path | str,
        var_set: VariableSet = None,
    ):
        """
        Initialize loader with manifest path and optional VariableSet.
        
        Args:
            manifest_path: Path to manifest JSON file
            var_set: Optional VariableSet for injection
        """
        self.manifest_path = Path(manifest_path)
        self.var_set = var_set
        self._template = load_configs_from_manifest(manifest_path)
        self.configs = copy.deepcopy(self._template)
    
    @property
    def keys(self) -> List[str]:
        """List of config keys loaded from manifest."""
        return list(self._template.keys())
    
    def reset(self) -> None:
        """Reset configs to original template values."""
        self.configs = copy.deepcopy(self._template)
    
    def inject(self, values: Dict[str, float]) -> Dict[str, Any]:
        """
        Inject values into configs and return the modified configs.
        
        Args:
            values: Dict mapping variable names to values
            
        Returns:
            Modified configs dict
        """
        if self.var_set is None:
            raise ValueError("No VariableSet provided for injection")
        
        self.configs = inject_from_varset(self._template, self.var_set, values)
        return self.configs
    
    def inject_case(self, case: List[float], var_names: List[str] = None) -> Dict[str, Any]:
        """
        Inject a single case (list of values) into configs.
        
        Args:
            case: List of values to inject
            var_names: Optional variable names. If None, uses var_set order.
            
        Returns:
            Modified configs dict
        """
        if self.var_set is None:
            raise ValueError("No VariableSet provided for injection")
        
        if var_names is None:
            var_names = [v.name for v in self.var_set.variables]
        
        values = dict(zip(var_names, case))
        return self.inject(values)
    
    def get(self, key: str) -> Any:
        """Get a specific config by key."""
        return self.configs.get(key)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access: loader["env"]."""
        return self.configs[key]



