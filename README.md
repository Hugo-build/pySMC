# pySMC

**Probabilistic Sampling, Monte Carlo, and Surrogate Modeling Utilities**

A Python library for probabilistic sampling, Monte Carlo simulation, and surrogate modeling with built-in support for adaptive sampling strategies (adaptive kriging, active learning, Bayesian optimization).

## Features

- 🎯 **Monte Carlo Sampling**: Multiple sampling strategies (random, LHS, Sobol sequences)
- 🔬 **Surrogate Modeling**: JAX-powered Gaussian Process surrogates with multiple kernel types
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

### Basic Gaussian Process Surrogate

```python
from core.GPax import GaussianProcess, RBF
from core.Variables import VariableSet, Variable
from core.Samplers import sample_inputs
import jax.numpy as jnp

# Define variables
vset = VariableSet([
    Variable(name="x1", kind="uniform", params={"low": 0.0, "high": 1.0}),
    Variable(name="x2", kind="uniform", params={"low": 0.0, "high": 1.0})
])

# Generate training data
X_train = sample_inputs(vset, 100, kind="lhs", seed=42)
y_train = your_function(X_train)  # Your function here

# Fit GP
kernel = RBF(log_sf=jnp.log(1.0), log_ls=jnp.log(jnp.ones(2) * 0.1))
gp = GaussianProcess(kernel=kernel, log_sn2=jnp.log(0.01), jitter=1e-6)
gp_fitted = gp.fit(jnp.array(X_train), jnp.array(y_train))

# Predict
X_test = sample_inputs(vset, 50, kind="sobol", seed=123)
y_pred, y_std = gp_fitted.predict(jnp.array(X_test))
```

### Adaptive Surrogate Modeling

```
Wait to be accomplished
```

### Monte Carlo Simulation via CLI

```bash
# List available case studies
python -m cli.main list-cases

# Run a case study
python -m cli.main run <case-name> --n 1000 --sampler sobol

# Run from config file
python -m cli.main run-config config.json --out results.csv
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
│   └── DataWash.py   # Data preprocessing
├── templates/         # Case study templates
├── cli/              # Command-line interface
├── io/               # Input/output utilities
└── docs/             # Documentation
```

## Key Components

### Surrogate Modeling Framework

The `Surrogates.py` module provides a universal adapter framework:

- **BaseSurrogate**: Abstract interface for all surrogate models
- **AdaptiveSurrogate**: Adds adaptive sampling capabilities
- **Acquisition Functions**: VarianceReduction, ExpectedImprovement, UCB
- **Model Adapters**: GaussianProcessAdapter, (future: Polynomial, NeuralNetwork)

See `docs/SURROGATE_DESIGN.md` for detailed architecture documentation.

### Gaussian Process Implementation

The `GPax.py` module provides a pure JAX implementation:

- Multiple kernel types: RBF, Matern32, Matern52
- Hyperparameter optimization with Optax
- Efficient prediction with uncertainty quantification
- Functional, immutable design

### Sampling Strategies

Available samplers in `Samplers.py`:

- `random`: Uniform random sampling
- `lhs`: Latin Hypercube Sampling
- `sobol`: Sobol sequences (quasi-random)

## Examples

See the following example scripts:

- `test_sobol_simple.py` - Sobol G-function example with GP
- `test_morris_simple.py` - Morris method example
- `test_sobolG_verbose.py` - Detailed Sobol analysis

## Documentation

- `docs/ARCHITECTURE_SUMMARY.txt` - High-level architecture overview
- `docs/SURROGATE_DESIGN.md` - Surrogate framework design
- `docs/GP_IMPROVEMENTS.md` - GP implementation details
- `docs/OPTIMIZER_GUIDE.md` - Optimizer configuration guide
- `docs/NUMERICAL_STABILITY.md` - Numerical stability considerations

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

## TODO

See `TODO.md` for planned features and improvements.

