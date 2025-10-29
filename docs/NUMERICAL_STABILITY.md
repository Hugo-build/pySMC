# Numerical Stability Improvements for GP Optimization

## Problem

During hyperparameter optimization, the loss function could become NaN, particularly with SGD optimizer. This was caused by:

1. **Hyperparameters diverging** to extreme values
2. **Exploding gradients** during backpropagation
3. **Ill-conditioned covariance matrices** leading to Cholesky decomposition failures
4. **Numerical overflow** in kernel computations

## Solution

Added three key numerical stability mechanisms:

### 1. Gradient Clipping

**What:** Clips gradients by global norm to prevent explosion.

**Where:** In `_clip_gradients()` and `make_standard_step()`

**How:**
```python
def _clip_gradients(grads, max_norm=10.0):
    # Compute global gradient norm
    total_norm = jnp.sqrt(sum(norm(g)^2 for g in grads))
    
    # Clip if exceeds threshold
    clip_factor = min(1.0, max_norm / (total_norm + 1e-8))
    
    return clip_factor * grads
```

**Effect:**
- Prevents gradient explosion that causes NaN
- Stable with SGD and other first-order methods
- Maintains gradient direction while constraining magnitude

### 2. Parameter Bounds

**What:** Clips log-scale hyperparameters to reasonable ranges.

**Where:** In `_clip_params()`

**Bounds:**
- `log_sf` ∈ [-6, 2] → signal variance σ_f ∈ [0.0025, 7.4]
- `log_ls` ∈ [-6, 2] → length scale ℓ ∈ [0.0025, 7.4]
- `log_sn2` ∈ [-8, 0] → noise std σ_n ∈ [3.4e-4, 1.0]

**Effect:**
- Prevents parameters from taking extreme values
- Keeps covariance matrix well-conditioned
- Maintains model expressiveness within practical range

### 3. NaN/Inf Checking

**What:** Detects and replaces NaN/Inf loss values with large penalty.

**Where:** In both `make_lbfgs_step()` and `make_standard_step()`

**How:**
```python
safe_loss = jnp.where(jnp.isfinite(loss), loss, 1e6)
```

**Effect:**
- Prevents optimization from getting stuck with NaN values
- Large penalty (1e6) discourages parameter configurations that cause NaN
- Allows optimizer to recover and find valid region

## Impact on Different Optimizers

### Adam Optimizer
- **Before:** Generally stable, rarely gets NaN
- **After:** More robust, faster convergence

### SGD Optimizer
- **Before:** Prone to NaN with default lr=0.01
- **After:** Much more stable with clipping

### LBFGS Optimizer
- **Before:** Very stable due to line search
- **After:** Even more robust with parameter bounds

## Testing

Tested with Sobol G-function in `test_sobolG.py`:
- ✓ 2D Sobol G with all optimizers
- ✓ 4D Sobol G with kernel comparison
- ✓ Different learning rates
- ✓ Edge cases with poor initialization

## Deployment

Changes made to `core/GP.py`:
1. Added `_clip_gradients()` function
2. Added `_clip_params()` function
3. Updated `make_lbfgs_step()` with safety checks
4. Updated `make_standard_step()` with gradient clipping
5. Added NaN/Inf detection in both step functions

No changes to user-facing API - all improvements are internal.
