# pySMC

**Probabilistic Sampling, Monte Carlo, and Surrogate Modeling Utilities**

A Python library for probabilistic sampling, Monte Carlo simulation, and surrogate modeling with built-in support for adaptive sampling strategies (adaptive kriging, active learning, Bayesian optimization).

## Features

- 🎯 **Monte Carlo Sampling**: Multiple sampling strategies (random, LHS, Sobol sequences)
- 🔬 **Surrogate Modeling**: JAX-powered Gaussian Process surrogates with multiple kernel types
- ✨ **sklearn-Style API**: Intuitive parameter interface (no manual log transforms!)
- 🔄 **Adaptive Sampling**: Built-in adaptive kriging, active learning, and Bayesian optimization
- 📊 **Design of Experiments**: Sobol G-function and other test functions
- 🛠️ **CLI Interface**: Command-line tools for Monte Carlo simulation
- 📦 **Case Studies**: Template-based framework for reproducible experiments

## Installation

### Prerequisites

- Python >= 3.10
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd pySMC

# Install with uv
uv pip install -e .

# Or install with pip
pip install -e .
```

### Dependencies

Core dependencies:
- `numpy>=1.26`
- `pandas>=2.1`
- `scipy>=1.11`
- `scikit-learn>=1.3`
- `matplotlib>=3.10.7`
- `jax>=0.6.2`
- `optax>=0.2.6`

Optional dev dependencies:
- `pytest>=8.4.2`
- `ruff>=0.6`

## Quick Start

> **Note**: pySMC now supports clean imports directly from the main package! 
> See [IMPORT_GUIDE.md](IMPORT_GUIDE.md) for detailed information about the new import structure.

### Basic Gaussian Process Surrogate

```python
# New clean imports from main package!
from pySMC import (
    GaussianProcess, RBF, optSetup,
    VariableSet, Variable,
    sample_inputs
)
import jax.numpy as jnp

# Define variables
vset = VariableSet([
    Variable(name="x1", kind="uniform", params={"low": 0.0, "high": 1.0}),
    Variable(name="x2", kind="uniform", params={"low": 0.0, "high": 1.0})
])

# Generate training data
X_train = sample_inputs(vset, 100, kind="lhs", seed=42)
y_train = your_function(X_train)  # Your function here

# Create GP with sklearn-style interface
kernel = RBF.from_params(
    signal_std=1.0,
    length_scale=jnp.ones(2) * 0.1
)
gp = GaussianProcess.from_params(
    kernel=kernel,
    noise_std=0.1
)

# Fit GP with hyperparameter optimization
opt_config = optSetup(optimizer='adam', steps=100, lr=0.01, verbose=True)
gp_fitted = gp.fit(jnp.array(X_train), jnp.array(y_train), opt_config=opt_config)

# Predict
X_test = sample_inputs(vset, 50, kind="sobol", seed=123)
y_pred, y_std = gp_fitted.predict(jnp.array(X_test))
```



### Surrogate Pipeline

The `SurrogatePipe` provides a higher-level abstraction with automatic data preprocessing:

```python
# New clean imports!
from pySMC import (
    SurrogatePipe, StandardScaler,
    GaussianProcess, RBF, optSetup,
    VariableSet, Variable,
    sample_inputs
)
import jax.numpy as jnp

# Define variables and generate data
vset = VariableSet([
    Variable(name="x1", kind="uniform", params={"low": 0.0, "high": 1.0}),
    Variable(name="x2", kind="uniform", params={"low": 0.0, "high": 1.0})
])
X_train = sample_inputs(vset, 100, kind="lhs", seed=42)
y_train = your_function(X_train)

# Create and fit scalers
x_scaler = StandardScaler()
y_scaler = StandardScaler()
x_scaler.fit(X_train)
y_scaler.fit(y_train.reshape(-1, 1))

# Scale training data
X_train_scaled = x_scaler.transform(X_train)
y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).flatten()

# Fit GP on scaled data
kernel = RBF.from_params(signal_std=1.0, length_scale=jnp.ones(2) * 0.2)
gp = GaussianProcess.from_params(kernel=kernel, noise_std=0.1)
opt_config = optSetup(optimizer='adam', steps=100, lr=0.02, verbose=True)
gp_fitted = gp.fit(jnp.array(X_train_scaled), jnp.array(y_train_scaled), opt_config=opt_config)

# Create surrogate pipe (handles scaling automatically)
pipe = SurrogatePipe(
    model=gp_fitted,
    varSet=vset,
    X=X_train,        # Original unscaled data
    y=y_train,        # Original unscaled data
    x_scaler=x_scaler,
    y_scaler=y_scaler
)

# Predict with automatic scaling/unscaling
predict_fn = pipe.make_predict_fn() # functional method
X_test = sample_inputs(vset, 50, kind="sobol", seed=123)
y_pred, y_std = predict_fn(X_test)  # Input/output in original scale
```

### Updatable Surrogate Modeling

Update surrogates with new data using adaptive weighting strategies:

```python
# New clean imports!
from pySMC import (
    SurrogatePipe, SurrogatePool, StandardScaler,
    calc_upd_weight, combine_weighted_data,
    SizeNoveltyWeight
)

# ... (assuming you have pipe, X_train, y_train from previous example)

# Generate new data (e.g., from adaptive sampling or new experiments)
X_new = sample_inputs(vset, 200, kind="lhs", seed=999)
y_new = your_function(X_new) # use your function of evaluation, getting corresponding y values.

# Calculate adaptive weight based on novelty and size
predict_fn = pipe.make_predict_fn()
weight = calc_upd_weight(
    X_old=X_train,
    y_old=y_train,
    X_new=X_new,
    y_new=y_new,
    predict_fn=predict_fn,
    strategy=SizeNoveltyWeight(novelty_power=0.5),
    verbose=True
)

# Combine datasets using weighted sampling
X_combined, y_combined = combine_weighted_data(
    X_old=X_train,
    y_old=y_train,
    X_new=X_new,
    y_new=y_new,
    weight=weight,
    random_state=42,
    verbose=True
)

# Fit new scalers and GP on combined data
x_scaler_new = StandardScaler().fit(X_combined)
y_scaler_new = StandardScaler().fit(y_combined.reshape(-1, 1))
X_combined_scaled = x_scaler_new.transform(X_combined)
y_combined_scaled = y_scaler_new.transform(y_combined.reshape(-1, 1)).flatten()

gp_updated = GaussianProcess.from_params(kernel=kernel, noise_std=0.1)
gp_updated_fitted = gp_updated.fit(
    jnp.array(X_combined_scaled), 
    jnp.array(y_combined_scaled), 
    opt_config=opt_config
)

# Create updated surrogate pipe
pipe_updated = SurrogatePipe(
    model=gp_updated_fitted,
    varSet=vset,
    X=X_combined,
    y=y_combined,
    x_scaler=x_scaler_new,
    y_scaler=y_scaler_new
)

# Manage multiple surrogates with SurrogatePool
pool = SurrogatePool(surrogates=[pipe, pipe_updated])
latest_surrogate = pool.get(-1)  # Get most recent surrogate
y_pred_updated, y_std_updated = latest_surrogate.make_predict_fn()(X_test)
```

### Monte Carlo Simulation via CLI

```bash
Wait to be accomplished
```

## Project Structure

```
pySMC/
├── core/              # Core functionality
│   ├── GPax.py       # JAX Gaussian Process implementation
│   ├── Surrogates.py # Surrogate modeling framework
│   ├── Samplers.py   # Sampling strategies
│   ├── DoEs.py       # Design of Experiments
│   ├── Variables.py  # Variable definitions
│   ├── MonteCarlo.py # Monte Carlo simulation
│   ├── DataWash.py   # Data preprocessing
│   ├── Weighted.py   # Adaptive weighting strategies
│   └── Aquiz.py      # Acquisition functions
├── examples/         # Example scripts and case studies
├── cli/              # Command-line interface
├── io/               # Input/output utilities
└── FElib.py          # Finite element library
```

## Key Components

### Surrogate Modeling Framework

The `Surrogates.py` module provides a universal surrogate pipeline framework:

- **SurrogatePipe**: Main surrogate model pipeline with automatic scaling and preprocessing
- **SurrogatePool**: Manages multiple surrogate models for ensemble and version tracking
- **StandardScaler**: Data normalization for inputs and outputs
- **Adaptive Learning Utilities**: `calc_upd_weight`, `combine_weighted_data` for updating surrogates
- **Type Conversion**: Seamless numpy/JAX interoperability with `to_numpy`

### Gaussian Process Implementation

The `GPax.py` module provides a pure JAX implementation:

- **Multiple kernel types**: RBF, Matern32, Matern52
- **sklearn-style parameter interface**: Direct parameter specification (no manual log transforms)
- **Hyperparameter optimization**: Integrated Optax optimizers (Adam, SGD, etc.)
- **Uncertainty quantification**: Efficient prediction with mean and standard deviation
- **Functional, immutable design**: All operations create new instances (frozen dataclasses)

### Sampling Strategies

Available samplers in `Samplers.py`:

- **`random`**: Uniform random sampling
- **`lhs`**: Latin Hypercube Sampling (space-filling design)
- **`sobol`**: Sobol sequences (quasi-random, low-discrepancy)

### Weighting Strategies

Available weighting strategies in `Weighted.py`:

- **`SizeNoveltyWeight`**: Balances old and new data based on dataset size and prediction novelty

## Examples

See the `examples/` directory for complete example scripts:

- **`test_surrogate_framework.py`** - Complete surrogate pipeline demonstration with adaptive learning
- **`test_sobol_simple.py`** - Sobol G-function example with Gaussian Process
- **`test_sobolG_verbose.py`** - Detailed Sobol sensitivity analysis
- **`test_morris_simple.py`** - Morris screening method example
- **`test_parametric_variables.py`** - Parametric variable studies
- **`test_variable_targets.py`** - Variable targeting and sensitivity
- **`exp_40barTruss.py`** - 40-bar truss finite element case study
- **`exp_dualOsicllator_degrade.py`** - Dual oscillator degradation analysis

## Development

### Setup Development Environment

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
ruff check .
ruff format .
```

## License

MIT License

## Authors

- Hugo

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Import Structure

pySMC now provides a clean, intuitive API. You can import everything directly from the main package:

```python
# Recommended: Direct imports from main package
from pySMC import (
    GaussianProcess, RBF,
    Variable, VariableSet,
    sample_inputs,
    SurrogatePipe,
)
```

The old import style still works for backward compatibility:

```python
# Legacy style (still works)
from pySMC.core.GPax import GaussianProcess, RBF
from pySMC.core.Variables import Variable, VariableSet
```

For more details, see [IMPORT_GUIDE.md](IMPORT_GUIDE.md).

## TODO

See `TODO.md` for planned features and improvements.

