"""
pySMC I/O Module

Input/output utilities for loading configurations, case definitions,
and parallel simulation execution.
"""

from .manifest import (
    load_configs_from_manifest,
    inject_from_varset,
    inject_cases,
    save_injected_config,
    ManifestLoader,
)

from .runner import (
    get_n_workers,
    RunnerConfig,
    ParallelRunner,
    SlurmJobRunner,
    SimpleFunctionRunner,
    run_parallel,
)

__all__ = [
    # Manifest loading
    "load_configs_from_manifest",
    "inject_from_varset",
    "inject_cases",
    "save_injected_config",
    "ManifestLoader",
    # Parallel runner
    "get_n_workers",
    "RunnerConfig",
    "ParallelRunner",
    "SlurmJobRunner",
    "SimpleFunctionRunner",
    "run_parallel",
]
