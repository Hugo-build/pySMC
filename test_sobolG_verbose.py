"""
Test Sobol G-function with pure JAX Gaussian Process (GP.py).

This demonstrates:
1. Using the Sobol G-function as a test problem
2. Training and testing with JAX GP
3. Comparing different kernels (RBF, Matérn 3/2, Matérn 5/2)
4. Hyperparameter optimization
5. Model evaluation metrics
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.GPax import GaussianProcess, RBF, Matern32, Matern52, optSetup
from core.DoEs import sobol_g


def generate_sobol_data(dim: int, n_samples: int, a_values: np.ndarray, seed: int = 42):
    """
    Generate training data for Sobol G-function.
    
    Args:
        dim: Dimensionality
        n_samples: Number of samples
        a_values: Importance parameters for Sobol G-function
        seed: Random seed
    
    Returns:
        X: Inputs (n_samples, dim) in [0, 1]^dim
        y: Outputs (n_samples,)
    """
    # Generate LHS-like samples using uniform random
    rng = np.random.RandomState(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, dim))
    
    # Evaluate Sobol G-function
    f = sobol_g(a_values)
    y = np.array([f(x)["y"] for x in X])
    
    return X, y


def evaluate_predictions(y_true, y_pred, y_std=None):
    """
    Compute evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_std: Predictive standard deviations (optional)
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy for metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Regression metrics
    r2 = 1.0 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    max_error = np.max(np.abs(y_true - y_pred))
    
    metrics = {
        'R²': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'Max Error': max_error,
    }
    
    # Add uncertainty metrics if available
    if y_std is not None:
        y_std = np.array(y_std)
        # Check calibration: are 95% of points within ±2σ?
        in_bounds = np.abs(y_true - y_pred) <= 2 * y_std
        calibration = np.mean(in_bounds) * 100
        avg_uncertainty = np.mean(y_std)
        
        metrics['Calibration (%)'] = calibration  # Should be ~95%
        metrics['Avg Uncertainty'] = avg_uncertainty
    
    return metrics


def test_sobol_basic():
    """Test 1: Basic Sobol G-function with 2D."""
    print("="*70)
    print("TEST 1: Basic 2D Sobol G-function")
    print("="*70)
    
    # Setup problem
    dim = 4
    a = np.array([1.0, 1.0, 1.0, 1.0])  # Equal importance
    n_train = 1000
    n_test = 200
    
    print(f"\nProblem setup:")
    print(f"  Dimensions: {dim}")
    print(f"  a parameters: {a}")
    print(f"  Training samples: {n_train}")
    print(f"  Test samples: {n_test}")
    
    # Generate data
    X_train, y_train = generate_sobol_data(dim, n_train, a, seed=42)
    X_test, y_test = generate_sobol_data(dim, n_test, a, seed=123)
    
    # Convert to JAX arrays
    X_train = jnp.array(X_train, dtype=jnp.float32)
    y_train = jnp.array(y_train, dtype=jnp.float32)
    X_test = jnp.array(X_test, dtype=jnp.float32)
    y_test = jnp.array(y_test, dtype=jnp.float32)
    
    # Initialize GP with Matérn 3/2 kernel
    kernel = Matern32(
        log_sf=jnp.log(jnp.std(y_train)),
        log_ls=jnp.log(jnp.ones(dim) * 0.3)
    )
    
    gp = GaussianProcess(
        kernel=kernel,
        log_sn2=jnp.log(jnp.array(1e-4)),
        jitter=1e-6
    )
    
    # Fit with hyperparameter optimization
    print("\nTraining GP with hyperparameter optimization...")
    opt_config = optSetup(
        optimizer='adam',
        steps=200,
        lr=0.05,
        verbose=True,
        log_every=50
    )
    
    gp_fitted = gp.fit(X_train, y_train, opt_config=opt_config)
    
    # Predict
    print("\nMaking predictions...")
    y_pred, y_std = gp_fitted.predict(X_test)
    
    # Evaluate
    metrics = evaluate_predictions(y_test, y_pred, y_std)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    for key, value in metrics.items():
        print(f"{key:20s}: {value:8.4f}")
    print("="*70)
    
    return gp_fitted, X_test, y_test, y_pred, y_std


def test_kernel_comparison():
    """Test 2: Compare different kernels."""
    print("\n" + "="*70)
    print("TEST 2: Kernel Comparison on 4D Sobol G-function")
    print("="*70)
    
    # Setup problem (more challenging)
    dim = 4
    a = np.array([0.5, 1.0, 2.0, 4.0])  # Different importances
    n_train = 200
    n_test = 100
    
    print(f"\nProblem setup:")
    print(f"  Dimensions: {dim}")
    print(f"  a parameters: {a}")
    print(f"  Training samples: {n_train}")
    
    # Generate data
    X_train, y_train = generate_sobol_data(dim, n_train, a, seed=42)
    X_test, y_test = generate_sobol_data(dim, n_test, a, seed=999)
    
    X_train = jnp.array(X_train, dtype=jnp.float32)
    y_train = jnp.array(y_train, dtype=jnp.float32)
    X_test = jnp.array(X_test, dtype=jnp.float32)
    y_test = jnp.array(y_test, dtype=jnp.float32)
    
    # Test different kernels
    kernels = {
        'RBF': RBF(
            log_sf=jnp.log(jnp.std(y_train)),
            log_ls=jnp.log(jnp.ones(dim) * 0.3)
        ),
        'Matérn 3/2': Matern32(
            log_sf=jnp.log(jnp.std(y_train)),
            log_ls=jnp.log(jnp.ones(dim) * 0.3)
        ),
        'Matérn 5/2': Matern52(
            log_sf=jnp.log(jnp.std(y_train)),
            log_ls=jnp.log(jnp.ones(dim) * 0.3)
        ),
    }
    
    results = {}
    
    for kernel_name, kernel in kernels.items():
        print(f"\n--- Testing {kernel_name} ---")
        
        gp = GaussianProcess(
            kernel=kernel,
            log_sn2=jnp.log(jnp.array(1e-4)),
            jitter=1e-6
        )
        
        # Fit with optimization
        opt_config = optSetup(
            optimizer='adam',
            steps=150,
            lr=0.05,
            verbose=False
        )
        
        gp_fitted = gp.fit(X_train, y_train, opt_config=opt_config)
        
        # Predict and evaluate
        y_pred, y_std = gp_fitted.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred, y_std)
        
        results[kernel_name] = metrics
        
        print(f"  R² = {metrics['R²']:.4f}, RMSE = {metrics['RMSE']:.4f}")
    
    # Print comparison
    print("\n" + "="*70)
    print("KERNEL COMPARISON RESULTS")
    print("="*70)
    print(f"{'Kernel':<15} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'Calib (%)':<10}")
    print("-"*70)
    for kernel_name, metrics in results.items():
        print(f"{kernel_name:<15} {metrics['R²']:<10.4f} {metrics['RMSE']:<10.4f} "
              f"{metrics['MAE']:<10.4f} {metrics.get('Calibration (%)', 0):<10.1f}")
    print("="*70)
    
    return results


def test_optimizer_comparison():
    """Test 3: Compare different optimizers."""
    print("\n" + "="*70)
    print("TEST 3: Optimizer Comparison")
    print("="*70)
    
    # Setup
    dim = 3
    a = np.array([1.0, 2.0, 3.0])
    n_train = 150
    
    X_train, y_train = generate_sobol_data(dim, n_train, a, seed=42)
    X_train = jnp.array(X_train, dtype=jnp.float32)
    y_train = jnp.array(y_train, dtype=jnp.float32)
    
    optimizers = {
        'Adam': ('adam', 200, 0.05),
        'SGD': ('sgd', 300, 0.001),
        'LBFGS': ('lbfgs', 50, 1.0),
    }
    
    for opt_name, (opt_type, steps, lr) in optimizers.items():
        print(f"\n--- Testing {opt_name} ---")
        
        kernel = Matern52(
            log_sf=jnp.log(jnp.std(y_train)),
            log_ls=jnp.log(jnp.ones(dim) * 0.3)
        )
        
        gp = GaussianProcess(kernel=kernel, log_sn2=jnp.log(jnp.array(1e-4)))
        
        opt_config = optSetup(
            optimizer=opt_type,
            steps=steps,
            lr=lr,
            verbose=True,
            log_every=steps // 5
        )
        
        gp_fitted = gp.fit(X_train, y_train, opt_config=opt_config)
        
        # Check NLML
        nlml = gp_fitted.neg_lml(X_train, y_train)
        print(f"  Final NLML: {nlml:.3f}")


def visualize_results(X_test, y_test, y_pred, y_std, title="Sobol G-function"):
    """Create visualization of results."""
    print("\nCreating visualizations...")
    
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    y_std = np.array(y_std)

    # ____Save the figure_________________________________
    from datetime import datetime
    from pathlib import Path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saveDir = Path("figs")
    saveDir.mkdir(parents=True, exist_ok=True)
    figPath = saveDir / f'sobolG_{timestamp}.png'
    plt.savefig(figPath, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved as '{figPath}'")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Predictions vs True (sequential)
    ax1 = axes[0]
    idx = np.arange(len(y_test))
    ax1.scatter(idx, y_test, c='blue', s=30, alpha=0.6, label='True', zorder=3)
    ax1.scatter(idx, y_pred, c='red', s=20, alpha=0.8, label='Predicted', zorder=2)
    ax1.fill_between(idx, y_pred - 2*y_std, y_pred + 2*y_std, 
                     alpha=0.2, color='red', label='±2σ', zorder=1)
    ax1.set_xlabel('Test Sample Index', fontsize=11)
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title('Sequential Predictions', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    ax2 = axes[1]
    ax2.scatter(y_test, y_pred, alpha=0.6, s=40)
    lim_min = min(y_test.min(), y_pred.min())
    lim_max = max(y_test.max(), y_pred.max())
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=2, label='Perfect fit')
    ax2.set_xlabel('True Values', fontsize=11)
    ax2.set_ylabel('Predicted Values', fontsize=11)
    ax2.set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # Plot 3: Error distribution
    ax3 = axes[2]
    errors = y_test - y_pred
    ax3.hist(errors, bins=20, alpha=0.7, edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax3.set_xlabel('Prediction Error', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} - GP Regression Results', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('test_sobolG_results.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved as 'test_sobolG_results.png'")
    plt.show()
    


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SOBOL G-FUNCTION TEST WITH PURE JAX GP")
    print("="*70)
    
    # Run tests
    gp_fitted, X_test, y_test, y_pred, y_std = test_sobol_basic()
    test_kernel_comparison()
    test_optimizer_comparison()
    
    # Visualize
    visualize_results(X_test, y_test, y_pred, y_std, title="2D Sobol G-function")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")

