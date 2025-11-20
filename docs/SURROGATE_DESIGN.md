# Surrogate Framework Design

## Overview

The `Surrogates.py` module provides a **universal adapter framework** for surrogate modeling with built-in support for **adaptive sampling strategies** (adaptive kriging, active learning, etc.).

## Design Philosophy

### 1. Universal Adapter Pattern
- Surrogates are **adapters**, not implementations
- Concrete models (GP, NN, polynomials) live in their own modules
- The framework provides a unified interface

### 2. Separation of Concerns
- `GP.py` remains **independent** and **functional** (immutable, JAX-native)
- Can use GP.py standalone OR through the surrogate framework
- Easy to add new model types without touching existing code

### 3. Extensibility
- Add new models by creating adapters
- Add new acquisition functions by extending `AcquisitionFunction`
- Mix and match components

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Surrogate Framework                      │
└─────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐
│  BaseSurrogate   │         │  AcquisitionFn   │
│      (ABC)       │         │      (ABC)       │
├──────────────────┤         ├──────────────────┤
│ + fit()          │         │ + evaluate()     │
│ + predict()      │         └──────────────────┘
│ + is_fitted()    │                │
└────────┬─────────┘                ├─ VarianceReduction
         │                          ├─ ExpectedImprovement
         │                          └─ UpperConfidenceBound
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌────────┐  ┌───────────────┐
│ Basic  │  │   Adaptive    │
│Surrog. │  │   Surrogate   │
└────────┘  └───────────────┘
    │              │
    │              ├─ suggest_next_sample()
    │              └─ add_sample()
    │
    ├─ GaussianProcessAdapter ──► wraps GP.py
    ├─ PolynomialAdapter (future)
    └─ NeuralNetworkAdapter (future)
```

---

## Core Components

### 1. BaseSurrogate (Abstract Base Class)

**Purpose:** Define the universal interface for all surrogates

**Methods:**
- `fit(X, y, **kwargs)` → Fitted surrogate
- `predict(X_star)` → (mean, std)
- `is_fitted()` → bool
- `validate_inputs(X, y)` → (X_validated, y_validated)

**Example:**
```python
from core.Surrogates import BaseSurrogate

class MyCustomAdapter(BaseSurrogate):
    def fit(self, X, y, **kwargs):
        # Fit your model
        return self
    
    def predict(self, X_star):
        # Return (mean, std)
        return mean, std
    
    def is_fitted(self):
        return self._fitted
```

### 2. AdaptiveSurrogate (Extends BaseSurrogate)

**Purpose:** Add adaptive sampling capabilities

**Additional Methods:**
- `suggest_next_sample(X_candidates, acquisition, n_samples)` → X_next
- `add_sample(X_new, y_new, refit=True)` → Updated surrogate
- `get_training_data()` → (X_train, y_train)

**Use Cases:**
- Adaptive kriging
- Active learning
- Sequential experimental design
- Bayesian optimization

### 3. Acquisition Functions

**Purpose:** Determine where to sample next

**Available Functions:**

#### VarianceReduction
- **Strategy:** Explore regions with high uncertainty
- **Use case:** Space-filling, global exploration
- **Formula:** `score = σ²`

```python
from core.Surrogates import VarianceReduction

acq = VarianceReduction()
```

#### ExpectedImprovement
- **Strategy:** Balance exploration and exploitation
- **Use case:** Optimization (finding min/max)
- **Formula:** `EI = (f_best - μ - ξ) * Φ(Z) + σ * φ(Z)`

```python
from core.Surrogates import ExpectedImprovement

acq = ExpectedImprovement(xi=0.01)  # xi controls exploration
```

#### UpperConfidenceBound
- **Strategy:** Optimistic exploration
- **Use case:** Optimization with controlled exploration
- **Formula:** `UCB = μ + β * σ` (for maximization)

```python
from core.Surrogates import UpperConfidenceBound

acq = UpperConfidenceBound(beta=2.0)  # beta controls exploration
```

### 4. Model Adapters

#### GaussianProcessAdapter

Wraps the JAX Gaussian Process from `GP.py`.

**Features:**
- Automatic hyperparameter initialization
- Multiple kernel types (RBF, Matérn 3/2, Matérn 5/2)
- Hyperparameter optimization
- Negative log marginal likelihood computation

**Example:**
```python
from core.Surrogates import GaussianProcessAdapter

# Create
gp = GaussianProcessAdapter.create(kernel_type='matern52')

# Fit with optimization
gp = gp.fit(X_train, y_train, 
           optimize_hyperparameters=True,
           optimizer='adam',
           steps=200,
           lr=0.05)

# Predict
mean, std = gp.predict(X_test)

# Check quality
nlml = gp.neg_log_marginal_likelihood(X_train, y_train)
```

#### AdaptiveGaussianProcess

Combines GP with adaptive sampling.

**Example:**
```python
from core.Surrogates import AdaptiveGaussianProcess, ExpectedImprovement

# Create and fit
agp = AdaptiveGaussianProcess.create(kernel_type='rbf')
agp = agp.fit(X_initial, y_initial)

# Adaptive loop
for i in range(n_iterations):
    # Suggest where to sample next
    X_next = agp.suggest_next_sample(
        X_candidates,
        acquisition=ExpectedImprovement(),
        n_samples=1
    )
    
    # Evaluate expensive function
    y_next = expensive_function(X_next)
    
    # Add sample and refit
    agp = agp.add_sample(X_next, y_next, refit=True)
```

---

## Usage Examples

### Example 1: Basic GP Surrogate

```python
from core.Surrogates import create_surrogate
import jax.numpy as jnp

# Generate data
X = jnp.linspace(0, 5, 30).reshape(-1, 1)
y = jnp.sin(X).squeeze()

# Create and fit
surrogate = create_surrogate(model_type='gp', kernel_type='matern52')
surrogate = surrogate.fit(X, y, optimize_hyperparameters=True)

# Predict
X_test = jnp.linspace(0, 5, 100).reshape(-1, 1)
mean, std = surrogate.predict(X_test)
```

### Example 2: Adaptive Kriging

```python
from core.Surrogates import AdaptiveGaussianProcess, VarianceReduction

# Start with few samples
X_init = jnp.array([[0.0], [2.5], [5.0]])
y_init = expensive_function(X_init)

# Create adaptive GP
agp = AdaptiveGaussianProcess.create(kernel_type='matern32')
agp = agp.fit(X_init, y_init)

# Adaptive sampling loop
X_candidates = jnp.linspace(0, 5, 200).reshape(-1, 1)

for i in range(10):  # Add 10 more samples
    # Suggest next sample (high uncertainty regions)
    X_next = agp.suggest_next_sample(
        X_candidates,
        acquisition=VarianceReduction(),
        n_samples=1
    )
    
    # Evaluate
    y_next = expensive_function(X_next)
    
    # Update model
    agp = agp.add_sample(X_next, y_next, refit=True)

# Final prediction with refined model
mean, std = agp.predict(X_test)
```

### Example 3: Bayesian Optimization

```python
from core.Surrogates import AdaptiveGaussianProcess, ExpectedImprovement

# Optimize a black-box function
def objective(x):
    return -(x - 2)**2 + 5  # Max at x=2

# Initial samples
X_init = jnp.array([[0.0], [1.0], [3.0], [4.0]])
y_init = objective(X_init.squeeze())

# Setup
agp = AdaptiveGaussianProcess.create(kernel_type='rbf')
agp = agp.fit(X_init, y_init)

X_candidates = jnp.linspace(0, 5, 100).reshape(-1, 1)

# Optimization loop
for i in range(15):
    # Suggest next sample (balance exploration/exploitation)
    X_next = agp.suggest_next_sample(
        X_candidates,
        acquisition=ExpectedImprovement(xi=0.01),
        n_samples=1
    )
    
    y_next = objective(X_next.squeeze())
    agp = agp.add_sample(X_next, y_next, refit=True)

# Best found
X_train, y_train = agp.get_training_data()
best_idx = jnp.argmax(y_train)
print(f"Best x: {X_train[best_idx, 0]:.3f}")
print(f"Best y: {y_train[best_idx]:.3f}")
```

---

## Extending the Framework

### Adding a New Model Type

1. **Create an adapter** that implements `BaseSurrogate`:

```python
from core.Surrogates import BaseSurrogate

class PolynomialAdapter(BaseSurrogate):
    def __init__(self, degree=2):
        self.degree = degree
        self.coeffs = None
    
    def fit(self, X, y, **kwargs):
        X, y = self.validate_inputs(X, y, fit_mode=True)
        # Fit polynomial
        self.coeffs = jnp.polyfit(X.squeeze(), y, self.degree)
        return self
    
    def predict(self, X_star):
        X_star, _ = self.validate_inputs(X_star, fit_mode=False)
        mean = jnp.polyval(self.coeffs, X_star.squeeze())
        std = jnp.zeros_like(mean)  # No uncertainty
        return mean, std
    
    def is_fitted(self):
        return self.coeffs is not None
```

2. **Use it:**

```python
poly = PolynomialAdapter(degree=3)
poly = poly.fit(X_train, y_train)
mean, std = poly.predict(X_test)
```

### Adding a New Acquisition Function

```python
from core.Surrogates import AcquisitionFunction

class ProbabilityOfImprovement(AcquisitionFunction):
    def __init__(self, xi=0.01):
        self.xi = xi
    
    def evaluate(self, mean, std, **kwargs):
        from jax.scipy.stats import norm
        
        best_value = kwargs.get('best_value', jnp.min(mean))
        std = jnp.maximum(std, 1e-8)
        
        z = (best_value - mean - self.xi) / std
        return norm.cdf(z)
```

---

## Benefits of This Design

### ✅ Independence
- **GP.py is standalone** - Can be used without the framework
- **No circular dependencies** - Clean module structure
- **Functional programming** - Immutable, composable

### ✅ Flexibility
- **Easy to add models** - Just create an adapter
- **Mix and match** - Combine different acquisition functions with any model
- **Extensible** - Framework grows without breaking existing code

### ✅ Practicality
- **Adaptive methods built-in** - Adaptive kriging works out of the box
- **Real-world ready** - Handles common use cases (optimization, space-filling)
- **Well-documented** - Clear examples and usage patterns

### ✅ Performance
- **JAX-native** - JIT compilation, automatic differentiation
- **Efficient** - Minimal overhead from adapter pattern
- **Scalable** - Can handle large datasets

---

## Testing

Run comprehensive tests:

```bash
python test_surrogate_framework.py
```

This will:
1. Test basic GP adapter
2. Demonstrate adaptive kriging
3. Compare acquisition functions
4. Test batch sample suggestion
5. Generate visualization

---

## Future Extensions

### Potential Additions

1. **More Model Types:**
   - Neural network adapter
   - Random forest adapter
   - Polynomial chaos expansion

2. **More Acquisition Functions:**
   - Knowledge gradient
   - Probability of improvement
   - Thompson sampling

3. **Advanced Features:**
   - Multi-objective optimization
   - Constrained optimization
   - Multi-fidelity surrogates

4. **Utilities:**
   - Cross-validation
   - Model comparison metrics
   - Hyperparameter tuning

---

## Summary

The surrogate framework provides:

- 🎯 **Universal interface** for all surrogate models
- 🔄 **Adaptive sampling** built-in (adaptive kriging, active learning)
- 🧩 **Modular design** - easy to extend
- ⚡ **JAX-powered** - fast and differentiable
- 📦 **GP integration** - wraps your excellent GP.py implementation
- 🛠️ **Production-ready** - well-tested and documented

**Philosophy:** The framework is an **adapter layer**, not a replacement. Your core implementations (like GP.py) remain independent and functional, while the framework provides a unified interface and adaptive capabilities on top.

