"""
Test SurrogatePipe with basic usage (no scalers).

This demonstrates:
1. SurrogatePipe without StandardScaler (simpler workflow)
2. Using GPax GaussianProcess as backend model
3. Direct fit and predict on original scale
4. Prediction with uncertainty quantification

This is a simpler alternative to test_surrogate_framework.py that skips
the data preprocessing step, suitable for well-conditioned problems or
when you want to handle scaling yourself.
"""

import sys
from pathlib import Path
# Add parent directory to path for imports when running from examples/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp
import numpy as np
from pprint import pprint

from core.Surrogates import SurrogatePipe, to_numpy
from core.GPax import GaussianProcess, RBF, Matern52, optSetup
from core.DoEs import sobol_g
from core.Variables import VariableSet, Variable
from core.DataWash import train_test_split
from core.Samplers import sample_inputs


def print_metrics(y_true, y_pred, y_std=None):
    """Print prediction metrics."""
    r2 = 1.0 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    max_error = np.max(np.abs(y_true - y_pred))
    metrics = {
        'R²': float(r2),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE (%)': float(mape),
        'Max Error': float(max_error),
    }
    pprint(metrics)
    return metrics


def main():
    # -------------------------------------------------------
    # Test 1: Setup Sobol G-function test problem
    # -------------------------------------------------------
    print("=" * 70)
    print("Testing SurrogatePipe (Simple - No Scalers)")
    print("=" * 70)
    
    # Sobol G-function parameters (controls importance of each dimension)
    a = np.array([0.5, 1.0, 2.0])  # Different importance for 4D problem
    n_dim = a.size
    f = sobol_g(a)
    
    print(f"\nRunning Sobol G-function with a = {a}")
    print(f"Problem dimensionality: {n_dim}")
    print(f"Expected output range: ~[0, 1] (well-conditioned)")
    
    # Create variable set
    vset = VariableSet([
        Variable(name=f"x_{i+1}", kind="uniform", params={"low": 0.0, "high": 1.0})
        for i in range(n_dim)
    ])
    
    # -------------------------------------------------------
    # Test 2: Generate and split data
    # -------------------------------------------------------
    print("\n" + "="*70)
    print("1. Generating training and test data")
    print("="*70)
    
    X = sample_inputs(vset, 300, kind="lhs", seed=42)
    y = np.array([f(x)["y"] for x in X])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123
    )
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set:     {X_test.shape[0]} samples")
    print(f"   y_train range: [{y_train.min():.4f}, {y_train.max():.4f}]")
    
    # -------------------------------------------------------
    # Test 3: Fit GP directly (no scaling)
    # -------------------------------------------------------
    print("\n" + "="*70)
    print("2. Fitting Gaussian Process (no scaling)")
    print("="*70)
    
    # Create GP with appropriate hyperparameters
    kernel = RBF.from_params(
        signal_std=float(jnp.std(y_train)), 
        length_scale=jnp.ones(n_dim) * 0.2
    )
    gp = GaussianProcess.from_params(
        kernel=kernel, 
        noise_std=0.05,  # Small noise for well-conditioned problem
        jitter=1e-6
    )
    
    print(f"   Kernel: {type(kernel).__name__}")
    print(f"   Initial signal_std: {jnp.exp(kernel.log_sf):.4f}")
    print(f"   Initial length_scale: {jnp.exp(kernel.log_ls[0]):.4f}")
    print(f"   Noise std: {jnp.sqrt(jnp.exp(gp.log_sn2)):.4f}")
    
    # Fit GP with hyperparameter optimization
    opt_config = optSetup(
        optimizer='adam', 
        steps=300, 
        lr=0.02, 
        verbose=True, 
        log_every=50
    )
    
    print("\n   Optimizing hyperparameters...")
    gp_fitted = gp.fit(
        jnp.array(X_train), 
        jnp.array(y_train), 
        opt_config=opt_config
    )
    
    print(f"\n   Optimized signal_std: {jnp.exp(gp_fitted.kernel.log_sf):.4f}")
    print(f"   Optimized length_scale: {jnp.exp(gp_fitted.kernel.log_ls[0]):.4f}")
    print(f"   Optimized noise_std: {jnp.sqrt(jnp.exp(gp_fitted.log_sn2)):.4f}")
    
    # -------------------------------------------------------
    # Test 4: Create SurrogatePipe (no scalers)
    # -------------------------------------------------------
    print("\n" + "="*70)
    print("3. Creating SurrogatePipe (no scalers)")
    print("="*70)
    
    pipe = SurrogatePipe(
        model=gp_fitted,
        varSet=vset,
        X=X_train,
        y=y_train
    )
    
    print(f"   ✓ SurrogatePipe created")
    print(f"   ✓ Scaling enabled: X={pipe._scaled4X}, y={pipe._scaled4y}")
    print(f"   ✓ Model type: {type(pipe.model).__name__}")
    print(f"   ✓ Training data: {pipe.X.shape[0]} samples, {pipe.X.shape[1]} dimensions")
    
    # -------------------------------------------------------
    # Test 5: Make predictions
    # -------------------------------------------------------
    print("\n" + "="*70)
    print("4. Making predictions on test set")
    print("="*70)
    
    # Create prediction function
    predict_fn = pipe.make_predict_fn()
    y_pred, y_std = predict_fn(X_test)
    
    # Convert to numpy for metrics
    y_pred_np = to_numpy(y_pred)
    y_std_np = to_numpy(y_std)
    
    print("\n   Prediction metrics:")
    metrics = print_metrics(y_test, y_pred_np, y_std_np)
    print(f"\n   Mean prediction uncertainty: {np.mean(y_std_np):.4f}")
    print(f"   Max prediction uncertainty:  {np.max(y_std_np):.4f}")
    print(f"   Min prediction uncertainty:  {np.min(y_std_np):.4f}")
    
    # -------------------------------------------------------
    # Test 6: Evaluate at specific points
    # -------------------------------------------------------
    print("\n" + "="*70)
    print("5. Testing prediction at specific points")
    print("="*70)
    
    # Test at center point
    X_center = np.array([[0.5, 0.5, 0.5]])
    y_center_true = f(X_center[0])["y"]
    y_center_pred, y_center_std = predict_fn(X_center)
    
    print(f"\n   Center point {list(X_center)}:")
    print(f"   True value:      {y_center_true:.6f}")
    print(f"   Predicted:       {to_numpy(y_center_pred)[0]:.6f}")
    print(f"   Uncertainty:     {to_numpy(y_center_std)[0]:.6f}")
    print(f"   Absolute error:  {abs(y_center_true - to_numpy(y_center_pred)[0]):.6f}")
    
    # Test at corner point
    X_corner = np.array([[0.0, 0.0, 0.0]])
    y_corner_true = f(X_corner[0])["y"]
    y_corner_pred, y_corner_std = predict_fn(X_corner)
    
    print(f"\n   Corner point {list(X_corner)}:")
    print(f"   True value:      {y_corner_true:.6f}")
    print(f"   Predicted:       {to_numpy(y_corner_pred)[0]:.6f}")
    print(f"   Uncertainty:     {to_numpy(y_corner_std)[0]:.6f}")
    print(f"   Absolute error:  {abs(y_corner_true - to_numpy(y_corner_pred)[0]):.6f}")
    
    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"✓ SurrogatePipe created without scalers")
    print(f"✓ GP fitted with hyperparameter optimization")
    print(f"✓ Test R² score: {metrics['R²']:.4f}")
    print(f"✓ Mean absolute error: {metrics['MAE']:.4f}")
    print(f"✓ Predictions include uncertainty quantification")
    
    if metrics['R²'] > 0.90:
        print("\n🎉 Excellent surrogate model performance!")
    elif metrics['R²'] > 0.75:
        print("\n✅ Good surrogate model performance!")
    else:
        print("\n⚠️  Surrogate model may need improvement (consider more samples or scaling)")
    
    print("="*70)
    
    return {
        'pipe': pipe,
        'metrics': metrics,
        'gp_fitted': gp_fitted,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred_np,
        'y_std': y_std_np
    }


if __name__ == "__main__":
    results = main()

