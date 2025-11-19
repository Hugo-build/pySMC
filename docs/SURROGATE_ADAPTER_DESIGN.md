# Surrogate Framework Architecture Design

## Overview

This document describes the architecture for the universal surrogate modeling framework in `core/Surrogates.py`. The design enables seamless integration of different surrogate backends (GPax, sklearn, PyTorch, etc.) while maintaining a consistent user interface.

## Design Principles

### 1. **Adapter Pattern**
- Surrogates are **adapters**, not implementations
- Concrete models (GP, NN, etc.) live in their own modules (e.g., `core/GPax.py`)
- Adapters wrap backends and provide a unified interface
- Backends remain independent and can be used standalone

### 2. **Data Type Flexibility**
- **Input**: Accepts numpy arrays, JAX arrays, or PyTorch tensors
- **Output**: Always returns JAX arrays for consistency
- **Internal**: Adapters convert to backend's preferred format internally
- Users don't need to worry about array types

### 3. **Progressive Enhancement**
- `BaseSurrogate`: Core interface (fit, predict, is_fitted)
- `AdaptiveSurrogate`: Adds adaptive sampling (suggest_next_sample, add_sample)
- Not all backends support all features (graceful degradation)

### 4. **Extensibility**
- Easy to add new backends (create adapter class)
- Easy to add new acquisition functions (extend AcquisitionFunction)
- Mix and match components

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    User Code                                │
│  (Uses numpy/jax/torch arrays interchangeably)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Surrogate Framework (Surrogates.py)            │
│                                                             │
│  ┌──────────────────────┐    ┌──────────────────────┐    │
│  │   BaseSurrogate      │    │ AcquisitionFunction   │    │
│  │   (ABC)              │    │   (ABC)               │    │
│  │                      │    │                       │    │
│  │ + fit()              │    │ + evaluate()          │    │
│  │ + predict()          │    └───────────────────────┘    │
│  │ + is_fitted()        │           │                      │
│  │ + validate_inputs()  │           ├─ VarianceReduction   │
│  └──────────┬───────────┘           ├─ ExpectedImprovement │
│             │                       └─ UpperConfidenceBound │
│             │                                                │
│             ▼                                                │
│  ┌──────────────────────┐                                   │
│  │ AdaptiveSurrogate   │                                   │
│  │   (ABC)             │                                   │
│  │                      │                                   │
│  │ + suggest_next_...() │                                   │
│  │ + add_sample()       │                                   │
│  │ + get_training_data()│                                   │
│  └──────────┬───────────┘                                   │
│             │                                                │
│     ┌───────┴───────┬───────────────┐                      │
│     │               │               │                       │
│     ▼               ▼               ▼                       │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐                  │
│  │GPax     │  │Sklearn   │  │Torch     │                  │
│  │Adapter  │  │Adapter   │  │Adapter   │                  │
│  │         │  │          │  │(future)  │                  │
│  │Full     │  │Basic     │  │          │                  │
│  │Adaptive │  │(no       │  │          │                  │
│  │Support  │  │adaptive) │  │          │                  │
│  └────┬────┘  └────┬─────┘  └────┬─────┘                  │
│       │            │             │                         │
└───────┼────────────┼─────────────┼─────────────────────────┘
        │            │             │
        ▼            ▼             ▼
┌───────────────────────────────────────────────────────────┐
│              Concrete Backend Implementations             │
│                                                           │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │GPax.py   │  │sklearn       │  │PyTorch       │      │
│  │(JAX)     │  │(numpy)       │  │(torch)       │      │
│  │          │  │              │  │              │      │
│  │+ fit()   │  │+ fit()       │  │+ fit()       │      │
│  │+ predict │  │+ predict()   │  │+ predict()   │      │
│  └──────────┘  └──────────────┘  └──────────────┘      │
└───────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Data Type Utilities

Located at the top of `Surrogates.py`:

```python
def to_numpy(array) -> np.ndarray:
    """Convert any array to numpy (handles jax, torch, numpy)"""
    
def to_jax(array) -> jnp.ndarray:
    """Convert any array to JAX (handles numpy, torch, jax)"""
    
def detect_array_type(array) -> str:
    """Detect backend type ('numpy', 'jax', 'torch')"""
```

**Why**: Allows users to pass arrays in any format. Adapters convert internally.

### 2. BaseSurrogate Interface

**Core Methods:**
- `fit(X, y, **kwargs)` → Fitted surrogate
- `predict(X_star)` → (mean, std) as JAX arrays
- `is_fitted()` → bool
- `validate_inputs(X, y)` → Validated JAX arrays

**Key Design Decisions:**
- Inputs accept `Union[np.ndarray, jnp.ndarray]` (can extend to torch)
- Outputs always JAX arrays for consistency
- `**kwargs` allows backend-specific options
- `validate_inputs()` is a static method for convenience

### 3. AdaptiveSurrogate Interface

**Additional Methods:**
- `suggest_next_sample(X_candidates, acquisition, n_samples)` → X_next
- `add_sample(X_new, y_new, refit=True)` → Updated surrogate
- `get_training_data()` → (X_train, y_train)

**Design Note:** Not all backends support adaptive features. sklearn GPs, for example, need full refitting. The adapter can still implement `AdaptiveSurrogate` but do full refits internally.

### 4. Acquisition Functions

**Current Implementations:**
- `VarianceReduction`: Select highest uncertainty
- `ExpectedImprovement`: Balance exploration/exploitation
- `UpperConfidenceBound`: Optimistic exploration

**Extensibility:** Easy to add new acquisition functions by extending `AcquisitionFunction`.

### 5. Concrete Adapters

#### GPaxAdapter (Full Adaptive Support)

**Features:**
- Wraps `core.GPax.GaussianProcess`
- Full adaptive sampling support
- Handles JAX arrays natively
- Supports GPax optimization configs

**Usage:**
```python
from core.Surrogates import GPaxAdapter
from core.GPax import RBF, optSetup

# Create adapter
adapter = GPaxAdapter(
    kernel=RBF(log_sf=..., log_ls=...),
    log_sn2=...
)

# Fit
adapter.fit(X_train, y_train, optimizer='adam', steps=100)

# Predict
mean, std = adapter.predict(X_test)

# Adaptive sampling
next_sample = adapter.suggest_next_sample(X_candidates)
adapter.add_sample(X_new, y_new, refit=True)
```

#### SklearnAdapter (Basic Support)

**Features:**
- Wraps `sklearn.gaussian_process.GaussianProcessRegressor`
- Converts numpy ↔ JAX internally
- Basic fit/predict only (no incremental updates)

**Usage:**
```python
from core.Surrogates import SklearnAdapter
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

adapter = SklearnAdapter(
    kernel=RBF() + WhiteKernel(),
    n_restarts_optimizer=10
)

adapter.fit(X_train, y_train)  # Accepts numpy or JAX
mean, std = adapter.predict(X_test)  # Returns JAX arrays
```

**Note:** sklearn doesn't support incremental updates. To add adaptive features, you'd need to refit the entire model each time (expensive but possible).

#### TorchAdapter (Future)

**Design Considerations:**
- Wrap PyTorch neural networks or GPs
- Handle device placement (CPU/GPU)
- Convert tensors ↔ JAX arrays
- Support both deterministic and probabilistic models

---

## Data Flow Examples

### Example 1: User passes numpy arrays

```python
import numpy as np
from core.Surrogates import GPaxAdapter

X_train = np.array([[1.0], [2.0], [3.0]])  # numpy
y_train = np.array([1.0, 4.0, 9.0])         # numpy

adapter = GPaxAdapter()
adapter.fit(X_train, y_train)  # Internally converts to JAX

mean, std = adapter.predict(X_test)  # Returns JAX arrays
mean_np = np.array(mean)  # User can convert back if needed
```

### Example 2: User passes JAX arrays

```python
import jax.numpy as jnp
from core.Surrogates import GPaxAdapter

X_train = jnp.array([[1.0], [2.0], [3.0]])  # JAX
y_train = jnp.array([1.0, 4.0, 9.0])        # JAX

adapter = GPaxAdapter()
adapter.fit(X_train, y_train)  # Works directly

mean, std = adapter.predict(X_test)  # Returns JAX arrays
```

### Example 3: Mixed backends

```python
import numpy as np
from core.Surrogates import GPaxAdapter, SklearnAdapter

# GPax adapter (JAX backend)
gpax = GPaxAdapter()
gpax.fit(X_train, y_train)  # Accepts numpy, uses JAX internally

# Sklearn adapter (numpy backend)
sklearn = SklearnAdapter()
sklearn.fit(X_train, y_train)  # Accepts numpy, uses numpy internally

# Both return JAX arrays for consistency
mean1, std1 = gpax.predict(X_test)
mean2, std2 = sklearn.predict(X_test)

# Both are JAX arrays, can be used interchangeably
```

---

## Factory Pattern (Optional)

For convenience, a factory function allows creating adapters by name:

```python
from core.Surrogates import create_surrogate

# Create GPax adapter
gp = create_surrogate('gpax', n_dim=2)

# Create sklearn adapter
skl = create_surrogate('sklearn', n_dim=2)

# Future: Create torch adapter
nn = create_surrogate('torch', model_type='neural_network')
```

---

## Extension Guide

### Adding a New Backend Adapter

1. **Create adapter class:**
```python
class MyBackendAdapter(BaseSurrogate):  # or AdaptiveSurrogate
    def __init__(self, **kwargs):
        super().__init__(name="MyBackendAdapter")
        # Initialize backend model
        
    def fit(self, X, y, **kwargs):
        # Convert inputs (use to_numpy() or to_jax())
        # Call backend's fit method
        # Set self._fitted = True
        return self
        
    def predict(self, X_star):
        # Convert inputs
        # Call backend's predict
        # Convert outputs to JAX
        return mean_jax, std_jax
        
    def is_fitted(self):
        return self._fitted
```

2. **Add to factory (optional):**
```python
def create_surrogate(backend='gpax', **kwargs):
    if backend == 'my_backend':
        return MyBackendAdapter(**kwargs)
    # ... existing code
```

### Adding a New Acquisition Function

```python
class MyAcquisition(AcquisitionFunction):
    def __init__(self, param=1.0):
        self.param = param
        
    def evaluate(self, mean, std, **kwargs):
        # Compute acquisition scores
        return scores  # JAX array
```

---

## Benefits of This Design

### ✅ **For Users:**
- **Simple API**: Same interface regardless of backend
- **Type Flexibility**: Don't worry about numpy vs JAX vs torch
- **Easy Switching**: Swap backends without changing code

### ✅ **For Developers:**
- **Modular**: Backends remain independent
- **Extensible**: Easy to add new adapters
- **Testable**: Each adapter can be tested independently

### ✅ **For the Framework:**
- **Future-Proof**: Easy to add new backends (torch, GPy, etc.)
- **Consistent**: Unified interface across all backends
- **Robust**: Handles missing dependencies gracefully

---

## Migration Path

### Current Code (Direct GPax Usage):
```python
from core.GPax import GaussianProcess, RBF

gp = GaussianProcess(kernel=RBF(...), log_sn2=...)
gp = gp.fit(X_train, y_train, opt_config=...)
mean, std = gp.predict(X_test)
```

### New Code (Using Adapter):
```python
from core.Surrogates import GPaxAdapter

adapter = GPaxAdapter(kernel=RBF(...), log_sn2=...)
adapter.fit(X_train, y_train, optimizer='adam', steps=100)
mean, std = adapter.predict(X_test)
```

**Benefits:**
- Can switch to sklearn easily: `SklearnAdapter(...)` instead
- Supports adaptive sampling out of the box
- Unified interface with other surrogates

---

## Testing Strategy

### Unit Tests:
- Test each adapter independently
- Test data type conversions
- Test error handling (missing dependencies, invalid inputs)

### Integration Tests:
- Test adapter with real backends
- Test adaptive sampling workflows
- Test factory pattern

### Example Test:
```python
def test_gpax_adapter():
    adapter = GPaxAdapter(n_dim=1)
    X = np.random.randn(10, 1)
    y = np.random.randn(10)
    
    # Test fit
    adapter.fit(X, y)
    assert adapter.is_fitted()
    
    # Test predict
    X_test = np.random.randn(5, 1)
    mean, std = adapter.predict(X_test)
    assert isinstance(mean, jnp.ndarray)
    assert mean.shape == (5,)
    
    # Test adaptive sampling
    X_candidates = np.random.randn(20, 1)
    next_sample = adapter.suggest_next_sample(X_candidates)
    assert next_sample.shape == (1, 1)
```

---

## Future Enhancements

1. **TorchAdapter**: Wrap PyTorch models (neural networks, GPs)
2. **GPyAdapter**: Wrap GPy library
3. **Multi-output Support**: Extend interface for vector-valued outputs
4. **Batch Prediction**: Optimize for large batches
5. **Model Persistence**: Save/load adapters
6. **Hyperparameter Tuning**: Built-in cross-validation

---

## Summary

The surrogate framework provides:
- ✅ **Universal Interface**: One API for all backends
- ✅ **Type Flexibility**: Accepts numpy/jax/torch seamlessly
- ✅ **Extensibility**: Easy to add new backends/functions
- ✅ **Adaptive Sampling**: Built-in support for active learning
- ✅ **Backend Independence**: Can use backends standalone or through framework

This design enables users to choose the best backend for their needs while maintaining a consistent, intuitive interface.

