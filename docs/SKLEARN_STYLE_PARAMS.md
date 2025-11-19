# sklearn-Style Parameter Interface for GPax

## Overview

The GPax module now provides a **sklearn-style interface** for initializing Gaussian Process models. This update eliminates the need for manual log transformations, making the API more intuitive while maintaining numerical stability internally.

## Motivation

### Before (Log-space parameters)
```python
# ❌ Old way - required manual log transforms
kernel = RBF(
    log_sf=jnp.log(jnp.std(y_train) + 1e-6),
    log_ls=jnp.log(jnp.ones((n_dim,)) * 0.1)
)
gp = GaussianProcess(
    kernel=kernel,
    log_sn2=jnp.log(jnp.array(0.1**2)),
    jitter=1e-6
)
```

**Problems:**
- Not intuitive - users must remember to take logarithms
- Inconsistent with sklearn API
- Error-prone (easy to forget `jnp.log()`)
- Requires understanding of internal log-space representation

### After (Direct parameters)
```python
# ✅ New way - direct parameters (sklearn-style)
kernel = RBF.from_params(
    signal_std=float(jnp.std(y_train)),
    length_scale=jnp.ones(n_dim) * 0.1
)
gp = GaussianProcess.from_params(
    kernel=kernel,
    noise_std=0.1,
    jitter=1e-6
)
```

**Benefits:**
- ✅ Intuitive - direct parameter specification
- ✅ Consistent with sklearn/MATLAB APIs
- ✅ Less error-prone
- ✅ Internally converts to log-space for stability

---

## API Reference

### Kernel Initialization

All kernels (`RBF`, `Matern32`, `Matern52`) now support the `from_params()` class method:

#### RBF (Squared Exponential) Kernel

```python
from core.GPax import RBF

# New interface (recommended)
kernel = RBF.from_params(
    signal_std=1.5,      # Signal standard deviation σ (positive)
    length_scale=0.5     # Length scale ℓ (positive, scalar or array)
)

# Old interface (still supported)
kernel = RBF(
    log_sf=jnp.log(jnp.array(1.5)),
    log_ls=jnp.log(jnp.array(0.5))
)
```

#### Matérn 3/2 Kernel

```python
from core.GPax import Matern32

kernel = Matern32.from_params(
    signal_std=1.5,
    length_scale=0.5
)
```

#### Matérn 5/2 Kernel

```python
from core.GPax import Matern52

kernel = Matern52.from_params(
    signal_std=1.5,
    length_scale=0.5
)
```

#### ARD (Automatic Relevance Determination)

For multi-dimensional inputs, specify a different length scale for each dimension:

```python
# ARD with 3 dimensions
length_scales = jnp.array([0.5, 1.0, 2.0])

kernel = RBF.from_params(
    signal_std=1.0,
    length_scale=length_scales  # Vector for ARD
)
```

### GaussianProcess Initialization

```python
from core.GPax import GaussianProcess

# New interface (recommended)
gp = GaussianProcess.from_params(
    kernel=kernel,
    noise_std=0.1,       # Observation noise standard deviation
    jitter=1e-6          # Numerical stability term (optional)
)

# Old interface (still supported)
gp = GaussianProcess(
    kernel=kernel,
    log_sn2=jnp.log(jnp.array(0.1**2)),
    jitter=1e-6
)
```

---

## Complete Workflow

### 1. Basic Example

```python
import jax.numpy as jnp
from core.GPax import RBF, GaussianProcess, optSetup

# Data
X_train = jnp.array([[0.1], [0.5], [0.9]])
y_train = jnp.array([0.2, 0.6, 0.8])

# Initialize with direct parameters
kernel = RBF.from_params(
    signal_std=1.0,
    length_scale=0.5
)

gp = GaussianProcess.from_params(
    kernel=kernel,
    noise_std=0.1
)

# Fit (no optimization)
gp_fitted = gp.fit(X_train, y_train, opt_config=None)

# Predict
X_test = jnp.array([[0.3], [0.7]])
y_pred, y_std = gp_fitted.predict(X_test)
```

### 2. With Hyperparameter Optimization

```python
# Setup optimization
opt_config = optSetup(
    optimizer='adam',     # 'adam', 'lbfgs', or 'sgd'
    steps=100,
    lr=0.01,
    verbose=True,
    log_every=10,
    # Convergence criteria (optional)
    tol_fun=1e-6,        # Stop if |Δloss| < tol_fun
    tol_x=1e-8,          # Stop if ||Δparams|| < tol_x
    tol_grad=1e-5,       # Stop if ||grad|| < tol_grad
    patience=20          # Early stopping patience
)

# Fit with optimization
gp_fitted = gp.fit(X_train, y_train, opt_config=opt_config)

# Make predictions
y_pred, y_std = gp_fitted.predict(X_test)
```

### 3. Data-Driven Initialization

Common pattern for initializing hyperparameters from data:

```python
# Estimate reasonable initial values from data
signal_std = float(jnp.std(y_train))
data_range = float(X_train.max() - X_train.min())
length_scale = data_range / 4.0  # Rule of thumb

kernel = RBF.from_params(
    signal_std=signal_std,
    length_scale=length_scale
)

gp = GaussianProcess.from_params(
    kernel=kernel,
    noise_std=0.1  # Or estimate from data noise level
)
```

### 4. Multi-dimensional with ARD

```python
# 4D problem
n_dim = 4
X_train = jnp.array([...])  # Shape: (N, 4)
y_train = jnp.array([...])  # Shape: (N,)

# Initialize with uniform length scales
kernel = RBF.from_params(
    signal_std=float(jnp.std(y_train)),
    length_scale=jnp.ones(n_dim) * 0.3
)

gp = GaussianProcess.from_params(
    kernel=kernel,
    noise_std=0.1
)

# Optimization will learn different length scales for each dimension
gp_fitted = gp.fit(X_train, y_train, opt_config=opt_config)

# Inspect learned length scales
params = gp_fitted.get_params_tree()
learned_ls = jnp.exp(params['kernel']['log_ls'])
print(f"Learned length scales: {learned_ls}")
```

---

## Comparison with sklearn

The new interface is similar to sklearn's `GaussianProcessRegressor`:

### sklearn
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

kernel = 1.0 * RBF(length_scale=0.5) + WhiteKernel(noise_level=0.1**2)
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(X_train, y_train)
y_pred, y_std = gp.predict(X_test, return_std=True)
```

### GPax (new interface)
```python
from core.GPax import RBF, GaussianProcess

kernel = RBF.from_params(signal_std=1.0, length_scale=0.5)
gp = GaussianProcess.from_params(kernel=kernel, noise_std=0.1)
gp_fitted = gp.fit(X_train, y_train, opt_config=None)
y_pred, y_std = gp_fitted.predict(X_test)
```

---

## Migration Guide

### Updating Existing Code

**Before:**
```python
kernel = RBF(log_sf=jnp.log(jnp.std(y_train)), log_ls=jnp.log(jnp.ones(n_dim) * 0.1))
gp = GaussianProcess(kernel=kernel, log_sn2=jnp.log(jnp.array(0.1**2)), jitter=1e-6)
```

**After:**
```python
kernel = RBF.from_params(signal_std=float(jnp.std(y_train)), length_scale=jnp.ones(n_dim) * 0.1)
gp = GaussianProcess.from_params(kernel=kernel, noise_std=0.1, jitter=1e-6)
```

### Pattern Replacements

| Old Pattern | New Pattern |
|------------|-------------|
| `log_sf=jnp.log(σ)` | `signal_std=σ` |
| `log_ls=jnp.log(ℓ)` | `length_scale=ℓ` |
| `log_sn2=jnp.log(σ_n²)` | `noise_std=σ_n` |

---

## Advanced Usage

### Accessing Internal Parameters

If you need to access the internal log-space parameters:

```python
gp = GaussianProcess.from_params(kernel=kernel, noise_std=0.1)
params = gp.get_params_tree()

# Access log-space parameters
log_sf = params['kernel']['log_sf']
log_ls = params['kernel']['log_ls']
log_sn2 = params['log_sn2']

# Convert back to direct parameters
signal_std = jnp.exp(log_sf)
length_scale = jnp.exp(log_ls)
noise_std = jnp.sqrt(jnp.exp(log_sn2))
```

### Mixing Interfaces

You can mix both interfaces if needed:

```python
# Create with new interface
gp = GaussianProcess.from_params(kernel=kernel, noise_std=0.1)

# Update with old interface (for advanced control)
new_kernel = RBF(log_sf=jnp.log(2.0), log_ls=jnp.log(1.5))
gp_updated = replace(gp, kernel=new_kernel)
```

---

## Implementation Details

### Why Log-space Internally?

The parameters are stored in log-space internally for several reasons:

1. **Unconstrained Optimization**: Log-space maps positive values to (-∞, +∞)
2. **Numerical Stability**: Prevents negative or zero values during optimization
3. **Better Gradients**: More stable gradient behavior
4. **Standard Practice**: Used in GPyTorch, GPflow, sklearn, MATLAB's fitrgp

The `from_params()` methods handle the conversion automatically:

```python
@classmethod
def from_params(cls, signal_std: float, length_scale: float):
    return cls(
        log_sf=jnp.log(jnp.asarray(signal_std)),
        log_ls=jnp.log(jnp.asarray(length_scale))
    )
```

---

## Examples Updated

All examples in the repository have been updated to use the new interface:

- ✅ `core/GPax.py` (main module with demo)
- ✅ `examples/test_sobol_simple.py`
- ✅ `examples/test_morris_simple.py`
- ✅ `examples/test_sobolG_verbose.py`
- ✅ `examples/exp_40barTruss.py`
- ✅ `examples/demo_new_interface.py` (new comprehensive demo)

---

## Summary

### Key Changes

1. **New Methods**:
   - `RBF.from_params(signal_std, length_scale)`
   - `Matern32.from_params(signal_std, length_scale)`
   - `Matern52.from_params(signal_std, length_scale)`
   - `GaussianProcess.from_params(kernel, noise_std, jitter)`

2. **Backward Compatibility**:
   - Old interface still works
   - Internal representation unchanged
   - No breaking changes

3. **Benefits**:
   - More intuitive API
   - Consistent with sklearn
   - Less error-prone
   - Better user experience

### Recommended Usage

```python
# ✅ RECOMMENDED: Use from_params() for clarity
kernel = RBF.from_params(signal_std=1.0, length_scale=0.5)
gp = GaussianProcess.from_params(kernel=kernel, noise_std=0.1)

# ⚠️ OLD: Still works, but not recommended
kernel = RBF(log_sf=jnp.log(1.0), log_ls=jnp.log(0.5))
gp = GaussianProcess(kernel=kernel, log_sn2=jnp.log(0.01))
```

---

## Questions?

See also:
- `examples/demo_new_interface.py` - Comprehensive demonstration
- `docs/GP_IMPROVEMENTS.md` - GP module improvements
- `docs/OPTIMIZER_GUIDE.md` - Optimization setup guide

