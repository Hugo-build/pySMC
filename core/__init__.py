"""
pySMC Core Module

This module contains the core functionality for surrogate modeling, 
adaptive sampling, and uncertainty quantification.
"""

# Surrogate modeling framework
from .Surrogates import (
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
from .GPax import (
    GaussianProcess,
    RBF,
    Matern32,
    Matern52,
    optSetup,
    get_optimizer,
)

# Variables and parameter definitions
from .Variables import (
    Variable,
    VariableSet,
    inject_single_config,
)

# Sampling strategies
from .Samplers import sample_inputs

# Design of Experiments
from .DoEs import (
    sobol_g,
    morris_g,
)

# Data preprocessing utilities
from .DataWash import (
    remove_zeros,
    remove_nan,
    train_test_split,
    train_test_validate_split,
    scale_data,
    unscale_data,
)

# Adaptive weighting strategies
from .Weighted import (
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
from .Aquiz import (
    AcquizFunc,
    VarianceMin,
    ExpectedImprovement,
    UpperConfidenceBound,
)

# Safe expression evaluation
from .SafeEval import evaluate_expression

__all__ = [
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


