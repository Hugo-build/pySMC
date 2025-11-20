# pySMC Import Guide

This guide explains the new import structure for pySMC after the `__init__.py` files have been properly configured.

## Overview

The pySMC package now provides a clean, intuitive API with three ways to import:

1. **Direct imports from main package** (RECOMMENDED)
2. **Imports from core submodule**
3. **Imports from specific modules** (legacy, still works)

## Quick Start

### Recommended: Import from Main Package

```python
# Import everything you need directly from pySMC
from pySMC import (
    GaussianProcess, RBF, Matern52,    # Gaussian Process models
    Variable, VariableSet,             # Variable definitions
    sample_inputs,                     # Sampling strategies
    SurrogatePipe, StandardScaler,     # Surrogate modeling
    train_test_split,                  # Data preprocessing
    sobol_g,                           # Test functions
)
```

## Available Exports

### Surrogate Modeling Framework

```python
from pySMC import (
    SurrogatePipe,          # Main surrogate model pipeline
    SurrogatePool,          # Manage multiple surrogates
    StandardScaler,         # Data normalization
    calc_upd_weight,        # Calculate update weights
    combine_weighted_data,  # Combine datasets with weights
    to_numpy,               # Convert to numpy arrays
    to_jax,                 # Convert to JAX arrays
    detect_array_type,      # Detect array backend
)
```

### Gaussian Process Implementation

```python
from pySMC import (
    GaussianProcess,    # Main GP class
    RBF,               # RBF/Squared Exponential kernel
    Matern32,          # Matern 3/2 kernel
    Matern52,          # Matern 5/2 kernel
    optSetup,          # Optimization configuration
    get_optimizer,     # Get optimizer by name
)
```

### Variables and Parameters

```python
from pySMC import (
    Variable,              # Single variable definition
    VariableSet,          # Collection of variables
    inject_single_config, # Inject config values
)
```

### Sampling Strategies

```python
from pySMC import sample_inputs

# Usage
X = sample_inputs(vset, n=100, kind="lhs", seed=42)
# Available kinds: "random", "lhs", "sobol"
```

### Design of Experiments

```python
from pySMC import (
    sobol_g,   # Sobol G-function
    morris_g,  # Morris function
)
```

### Data Preprocessing

```python
from pySMC import (
    remove_zeros,                # Remove zero entries
    remove_nan,                  # Remove NaN entries
    train_test_split,           # Split train/test
    train_test_validate_split,  # Split train/test/validate
    scale_data,                 # Scale data
    unscale_data,               # Unscale data
)
```

### Weighting Strategies

```python
from pySMC import (
    WeightStrategy,          # Base class
    UniformWeight,           # Uniform weighting
    SizeWeight,             # Size-based weighting
    NoveltyWeight,          # Novelty-based weighting
    SizeNoveltyWeight,      # Combined size + novelty
    UncertaintyBasedWeight, # Uncertainty-based weighting
    CustomWeightTemplate,   # Custom weighting template
    get_default_strategy,   # Get default strategy
    list_weight_strategies, # List all strategies
)
```

### Acquisition Functions

```python
from pySMC import (
    AcquizFunc,              # Base class
    VarianceMin,            # Minimize variance
    ExpectedImprovement,    # Expected improvement
    UpperConfidenceBound,   # Upper confidence bound
)
```

### Monte Carlo Simulation

```python
from pySMC import (
    MCResult,         # Monte Carlo result container
    run_monte_carlo,  # Run Monte Carlo simulation
)
```

### Utilities

```python
from pySMC import evaluate_expression  # Safe expression evaluation
```

## Migration Guide

If you have existing code using the old import style, here's how to migrate:

### Old Style (still works)

```python
from core.GPax import GaussianProcess, RBF
from core.Variables import Variable, VariableSet
from core.Samplers import sample_inputs
from core.Surrogates import SurrogatePipe
from core.DoEs import sobol_g
```

### New Style (recommended)

```python
from pySMC import (
    GaussianProcess, RBF,
    Variable, VariableSet,
    sample_inputs,
    SurrogatePipe,
    sobol_g,
)
```

### Alternative: Import from core submodule

```python
from pySMC.core import (
    GaussianProcess, RBF,
    Variable, VariableSet,
    sample_inputs,
    SurrogatePipe,
    sobol_g,
)
```

## Complete Example

Here's a complete example using the new import structure:

```python
import numpy as np
import jax.numpy as jnp

# New clean imports from main package
from pySMC import (
    GaussianProcess, RBF, optSetup,
    Variable, VariableSet,
    sample_inputs,
    SurrogatePipe, StandardScaler,
    train_test_split,
    sobol_g,
)

# Define variables
vset = VariableSet([
    Variable(name="x1", kind="uniform", params={"low": 0.0, "high": 1.0}),
    Variable(name="x2", kind="uniform", params={"low": 0.0, "high": 1.0}),
    Variable(name="x3", kind="uniform", params={"low": 0.0, "high": 1.0}),
])

# Generate training data using Sobol G-function
a = np.array([0.5, 1.0, 2.0])
f = sobol_g(a)
X = sample_inputs(vset, 200, kind="lhs", seed=42)
y = np.array([f(x)["y"] for x in X])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create and fit GP
kernel = RBF.from_params(
    signal_std=float(jnp.std(y_train)),
    length_scale=jnp.ones(3) * 0.2
)
gp = GaussianProcess.from_params(kernel=kernel, noise_std=0.05)

opt_config = optSetup(optimizer='adam', steps=200, lr=0.02, verbose=True)
gp_fitted = gp.fit(jnp.array(X_train), jnp.array(y_train), opt_config=opt_config)

# Create surrogate pipe
pipe = SurrogatePipe(model=gp_fitted, varSet=vset, X=X_train, y=y_train)

# Make predictions
predict_fn = pipe.make_predict_fn()
y_pred, y_std = predict_fn(X_test)

print(f"Test R²: {1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2):.4f}")
```

## Package Metadata

You can access package metadata:

```python
import pySMC

print(pySMC.__version__)  # "0.1.0"
print(pySMC.__author__)   # "yuma"
print(len(pySMC.__all__))  # Number of exported symbols
```

## Benefits of New Structure

1. **Cleaner imports**: No need to navigate deep module hierarchies
2. **Better discoverability**: All main components accessible from top level
3. **Backward compatible**: Old import style still works
4. **IDE-friendly**: Better autocomplete and type hints
5. **Consistent with best practices**: Similar to scikit-learn, pandas, etc.

## Notes

- The old import style (`from core.Module import ...`) still works for backward compatibility
- The `io/` and `cli/` modules are under refactoring and may have limited functionality
- All core functionality is fully accessible through the new import structure

