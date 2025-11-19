"""
Demonstration of the new sklearn-style parameter interface for GPax.

This script shows how to use the from_params() class methods for
intuitive, direct parameter initialization (no log transforms needed!).
"""

import jax.numpy as jnp
import numpy as np
from core.GPax import RBF, Matern32, Matern52, GaussianProcess, optSetup

# ==============================================================================
# OLD INTERFACE (still supported for advanced users)
# ==============================================================================
print("=" * 70)
print("OLD INTERFACE (log-space parameters)")
print("=" * 70)

# OLD: Had to manually compute log transforms
kernel_old = RBF(
    log_sf=jnp.log(jnp.array(1.5)),  # log of signal standard deviation
    log_ls=jnp.log(jnp.array(0.5))   # log of length scale
)

gp_old = GaussianProcess(
    kernel=kernel_old,
    log_sn2=jnp.log(jnp.array(0.1**2)),  # log of noise variance
    jitter=1e-6
)

print("✗ Old way: RBF(log_sf=jnp.log(1.5), log_ls=jnp.log(0.5))")
print("✗ Old way: GaussianProcess(kernel=k, log_sn2=jnp.log(0.1**2))")
print()

# ==============================================================================
# NEW INTERFACE (recommended - sklearn-style!)
# ==============================================================================
print("=" * 70)
print("NEW INTERFACE (direct parameters - sklearn-style!)")
print("=" * 70)

# NEW: Direct parameter specification - much cleaner!
kernel_new = RBF.from_params(
    signal_std=1.5,      # signal standard deviation (no log!)
    length_scale=0.5     # length scale (no log!)
)

gp_new = GaussianProcess.from_params(
    kernel=kernel_new,
    noise_std=0.1,       # noise standard deviation (no log!)
    jitter=1e-6
)

print("✓ New way: RBF.from_params(signal_std=1.5, length_scale=0.5)")
print("✓ New way: GaussianProcess.from_params(kernel=k, noise_std=0.1)")
print()

# ==============================================================================
# COMPLETE EXAMPLE
# ==============================================================================
print("=" * 70)
print("COMPLETE EXAMPLE: Training a GP on synthetic data")
print("=" * 70)

# Generate some synthetic data
np.random.seed(42)
X_train = np.linspace(0, 10, 20).reshape(-1, 1)
y_train = np.sin(X_train).squeeze() + 0.1 * np.random.randn(20)

# Convert to JAX arrays
X_train_jax = jnp.array(X_train)
y_train_jax = jnp.array(y_train)

# Estimate reasonable hyperparameters from data
signal_std = float(jnp.std(y_train_jax))
length_scale = float((X_train_jax.max() - X_train_jax.min()) / 4.0)
noise_std = 0.1

print(f"Data-driven initial hyperparameters:")
print(f"  - signal_std: {signal_std:.3f}")
print(f"  - length_scale: {length_scale:.3f}")
print(f"  - noise_std: {noise_std:.3f}")
print()

# ==============================================================================
# Different kernels (all with same clean interface!)
# ==============================================================================
print("All kernels use the same clean interface:")
print()

kernels = {
    'RBF (Squared Exponential)': RBF.from_params(
        signal_std=signal_std,
        length_scale=length_scale
    ),
    'Matérn 3/2': Matern32.from_params(
        signal_std=signal_std,
        length_scale=length_scale
    ),
    'Matérn 5/2': Matern52.from_params(
        signal_std=signal_std,
        length_scale=length_scale
    ),
}

for kernel_name, kernel in kernels.items():
    print(f"  ✓ {kernel_name}")
    
    # Create GP with this kernel
    gp = GaussianProcess.from_params(
        kernel=kernel,
        noise_std=noise_std,
        jitter=1e-6
    )
    
    # Note: We could fit it here if we wanted to
    # gp_fitted = gp.fit(X_train_jax, y_train_jax, opt_config=None)

print()

# ==============================================================================
# ARD (Automatic Relevance Determination) - Multi-dimensional length scales
# ==============================================================================
print("=" * 70)
print("ARD: Different length scales for each dimension")
print("=" * 70)

# For multi-dimensional input, you can specify different length scales
# for each dimension (Automatic Relevance Determination)
n_dims = 3
length_scales_ard = jnp.array([0.5, 1.0, 2.0])  # Different for each dimension

kernel_ard = RBF.from_params(
    signal_std=1.0,
    length_scale=length_scales_ard  # Vector of length scales
)

print(f"✓ ARD kernel with length_scale={length_scales_ard}")
print(f"  Dimension 1: ℓ = {length_scales_ard[0]}")
print(f"  Dimension 2: ℓ = {length_scales_ard[1]}")
print(f"  Dimension 3: ℓ = {length_scales_ard[2]}")
print()

# ==============================================================================
# OPTIMIZATION SETUP (unchanged - still clean and powerful!)
# ==============================================================================
print("=" * 70)
print("OPTIMIZATION (unchanged)")
print("=" * 70)

opt_config = optSetup(
    optimizer='adam',  # or 'lbfgs', 'sgd'
    steps=100,
    lr=0.01,
    verbose=True,
    log_every=10,
    # Convergence criteria (optional)
    tol_fun=1e-6,      # Function tolerance
    tol_x=1e-8,        # Parameter change tolerance
    tol_grad=1e-5,     # Gradient tolerance
    patience=20        # Early stopping patience
)

print("✓ optSetup with convergence criteria")
print()

# ==============================================================================
# SUMMARY
# ==============================================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("Key improvements:")
print("  1. ✓ Direct parameter specification (no log transforms!)")
print("  2. ✓ Consistent with sklearn API")
print("  3. ✓ More intuitive for users")
print("  4. ✓ Internally uses log-space for optimization stability")
print("  5. ✓ Old interface still works for advanced users")
print()
print("Usage pattern:")
print("  kernel = RBF.from_params(signal_std=1.0, length_scale=0.5)")
print("  gp = GaussianProcess.from_params(kernel=kernel, noise_std=0.1)")
print("  gp_fitted = gp.fit(X, y, opt_config=opt_config)")
print("  y_pred, y_std = gp_fitted.predict(X_test)")
print()
print("=" * 70)

