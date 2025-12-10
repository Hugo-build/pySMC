"""
pySMC: Probabilistic Sampling, Monte Carlo, and Surrogate Modeling Utilities

A Python library for probabilistic sampling, Monte Carlo simulation, and surrogate 
modeling with built-in support for adaptive sampling strategies.
"""

__version__ = "0.1.0"
__author__ = "yuma"
# =============================================================
#                     Import core modules
# =============================================================
# Core surrogate modeling framework
from .core.Surrogates import (
    SurrogatePipe,
    SurrogatePool,
    StandardScaler,
    calc_upd_weight,
    combine_weighted_data,
    to_numpy,
    to_jax,
    detect_array_type,
)

# Gaussian Process implementation
from .core.GPax import (
    GaussianProcess,
    RBF,
    Matern32,
    Matern52,
    optSetup,
    get_optimizer,
)

# Variables and parameter definitions
from .core.Variables import (
    Variable,
    VariableSet,
    inject_single_config,
)

# Sampling strategies
from .core.Samplers import sample_inputs

# Design of Experiments
from .core.DoEs import (
    sobol_g,
    morris_g,
)

# Data preprocessing utilities
from .core.DataWash import (
    remove_zeros,
    remove_nan,
    train_test_split,
    train_test_validate_split,
    scale_data,
    unscale_data,
)

# Adaptive weighting strategies
from .core.Weighted import (
    WeightStrategy,
    UniformWeight,
    SizeWeight,
    NoveltyWeight,
    SizeNoveltyWeight,
    UncertaintyBasedWeight,
    CustomWeightTemplate,
    get_default_strategy,
    list_all as list_weight_strategies,
)

# Acquisition functions for adaptive sampling
from .core.Aquiz import (
    AcquizFunc,
    VarianceMin,
    ExpectedImprovement,
    UpperConfidenceBound,
)

# Safe expression evaluation
from .core.SafeEval import evaluate_expression

# I/O utilities for manifest-based config loading
from .io.manifest import (
    load_configs_from_manifest,
    inject_from_varset,
    inject_cases,
    ManifestLoader,
)

# Parallel runner for batch simulations
from .io.runner import (
    ParallelRunner,
    SlurmJobRunner,
    SimpleFunctionRunner,
    RunnerConfig,
    run_parallel,
    get_n_workers,
)

#
# =============================================================
#                         Import dependent libraries
# =============================================================
from.lib import FElib





# =============================================================
#                         Define public API
# =============================================================
# Define public API
__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Surrogate modeling
    "SurrogatePipe",
    "SurrogatePool",
    "StandardScaler",
    "calc_upd_weight",
    "combine_weighted_data",
    "to_numpy",
    "to_jax",
    "detect_array_type",
    
    # Gaussian Process
    "GaussianProcess",
    "RBF",
    "Matern32",
    "Matern52",
    "optSetup",
    "get_optimizer",
    
    # Variables
    "Variable",
    "VariableSet",
    "inject_single_config",
    
    # I/O - Manifest loading
    "load_configs_from_manifest",
    "inject_from_varset",
    "inject_cases",
    "ManifestLoader",
    
    # I/O - Parallel runner
    "ParallelRunner",
    "SlurmJobRunner",
    "SimpleFunctionRunner",
    "RunnerConfig",
    "run_parallel",
    "get_n_workers",
    
    # Sampling
    "sample_inputs",
    
    # Design of Experiments
    "sobol_g",
    "morris_g",
    
    # Data preprocessing
    "remove_zeros",
    "remove_nan",
    "train_test_split",
    "train_test_validate_split",
    "scale_data",
    "unscale_data",
    
    # Weighting strategies
    "WeightStrategy",
    "UniformWeight",
    "SizeWeight",
    "NoveltyWeight",
    "SizeNoveltyWeight",
    "UncertaintyBasedWeight",
    "CustomWeightTemplate",
    "get_default_strategy",
    "list_weight_strategies",
    
    # Acquisition functions
    "AcquizFunc",
    "VarianceMin",
    "ExpectedImprovement",
    "UpperConfidenceBound",
    
    # Monte Carlo
    "MCResult",
    "run_monte_carlo",
    
    # Utilities
    "evaluate_expression",
]
