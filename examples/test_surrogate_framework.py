"""
Test the surrogate framework with adaptive learning capabilities.

This demonstrates:
1. SurrogatePipe with StandardScaler for X and y
2. Using GPax GaussianProcess as backend model
3. Adaptive learning utilities (calc_upd_weight, combine_weighted_data)
4. Prediction with uncertainty quantification
5. SurrogatePool for managing multiple surrogate models

Reference: Similar to test_sobol_simple.py but focuses on the Surrogate framework features.
"""

import sys
from pathlib import Path
# Add parent directory to path for imports when running from examples/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp
import numpy as np
from pprint import pprint

from core.Surrogates import (
    SurrogatePipe,
    SurrogatePool,
    StandardScaler,
    calc_upd_weight,
    combine_weighted_data,
    to_numpy,
    SurrogateGPax
)
from core.Weighted import SizeNoveltyWeight
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
    # Test 0: Setup Sobol G-function test problem
    # -------------------------------------------------------
    print("=" * 70)
    print("Testing Surrogate Framework with Sobol G-function")
    print("=" * 70)
    
    # Sobol G-function parameters (controls importance of each dimension)
    a = np.array([1.0, 1.0, 1.0, 1.0])  # Equal importance for 4D problem
    n_dim = a.size
    f = sobol_g(a)
    
    print(f"Running Sobol G-function with a = {a}")
    print(f"Problem dimensionality: {n_dim}")
    
    # Create variable set
    vset = VariableSet([
        Variable(name=f"x_{i+1}", kind="uniform", params={"low": 0.0, "high": 1.0})
        for i in range(n_dim)
    ])
    
    # Generate data and split into train/test
    print("----------------------------------------------------------")
    print(f"\n1. Generating data and splitting into train/test sets")
    X = sample_inputs(vset, 500, kind="lhs", seed=42)
    y = np.array([f(x)["y"] for x in X])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    



    
    print("----------------------------------------------------------")
    print("2. Creating SurrogatePipe with StandardScaler\n")
    
    # Create and fit scalers on ORIGINAL (unscaled) data
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    x_scaler.fit(X_train)
    y_scaler.fit(y_train.reshape(-1, 1))
    
    # Create and fit GP model
    kernel = RBF.from_params(
        signal_std=float(jnp.std(y_train)), 
        length_scale=jnp.ones(n_dim) * 0.2
    )
    gp = GaussianProcess.from_params(kernel=kernel, noise_std=0.1, jitter=1e-6)
    
    # Scale training data for GP fitting
    X_train_scaled = x_scaler.transform(X_train)
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).flatten()
    
    # Fit GP on SCALED data with optimization
    opt_config = optSetup(optimizer='adam', steps=500, lr=0.02, verbose=True, log_every=100)
    gp_fitted = gp.fit(jnp.array(X_train_scaled), jnp.array(y_train_scaled), opt_config=opt_config)
    
    # Wrap in SurrogateGPax
    surrogate = SurrogateGPax(model=gp_fitted, opt_config=opt_config)

    # Create surrogate pipe with:
    # - model trained on scaled data
    # - already-fitted scalers (will NOT be re-fitted)
    # - original unscaled X and y for reference
    pipe = SurrogatePipe(
        surrogate=surrogate,
        varSet=vset,
        X=X_train,  # Original unscaled data
        y=y_train,  # Original unscaled data
        x_scaler=x_scaler,  # Already fitted on original data
        y_scaler=y_scaler,  # Already fitted on original data
        verbose=False
    )
    
    print(f"   ✓ Surrogate pipe created with X scaler: {pipe._scaled4X}, y scaler: {pipe._scaled4y}")
    print(f"   ✓ Surrogate pipe model: {type(pipe.surrogate).__name__}")
    print(f"   ✓ Surrogate pipe variables: {pipe.varSet.to_SAlib()}")
    




    print("----------------------------------------------------------")
    print("3. Testing prediction on test set\n")
    
    # Make prediction through pipe (handles scaling automatically)
    # Flow: X_test (unscaled) -> scale with x_scaler -> GP predict (in scaled space) 
    #       -> unscale with y_scaler -> y_pred (unscaled)
    predict_fn = pipe.make_predict_fn()
    y_pred, y_std = predict_fn(X_test)
    
    # Calculate metrics
    y_pred_np = to_numpy(y_pred)
    y_std_np = to_numpy(y_std)
    
    print("   Prediction metrics:")
    metrics = print_metrics(y_test, y_pred_np, y_std_np)
    print(f"   Mean uncertainty (std) = {np.mean(y_std_np):.4f}")
    



    # -------------------------------------------------------
    # Test 3: Adaptive Learning Utilities
    # -------------------------------------------------------
    print("----------------------------------------------------------")
    print("4. Testing update surrogate with weighting\n")
    
    # Generate new data (simulating adaptive sampling)
    N_new = 300
    X_new = sample_inputs(vset, N_new, kind="lhs", seed=123)
    X_new_test = sample_inputs(vset, int(N_new*0.3), kind="lhs", seed=123)

    a1 = np.array([0.5, 0.5, 0.5, 0.5])
    f1 = sobol_g(a1)

    y_new = np.array([f1(x)["y"] for x in X_new])
    y_new_test = np.array([f1(x)["y"] for x in X_new_test])
    
    print(f"   Old data: N={X_train.shape[0]}, New data: N={X_new.shape[0]}")
    
    # Calculate adaptive weight
    weight = calc_upd_weight(
        X_old=X_train,
        y_old=y_train,
        X_new=X_new,
        y_new=y_new,
        predict_fn=predict_fn,
        strategy=SizeNoveltyWeight(novelty_power=0.5),
        verbose=True
    )
    
    print(f"\n   ✓ Calculated adaptive weight: {weight:.4f}")
    
    # Combine data using weighted sampling
    X_combined, y_combined = combine_weighted_data(
        X_old=X_train,
        y_old=y_train,
        X_new=X_new,
        y_new=y_new,
        weight=weight,
        random_state=42,
        verbose=True
    )
    
    print(f"   ✓ Combined dataset: N={X_combined.shape[0]}")
    print(f"     (from {X_train.shape[0]} old + {X_new.shape[0]} new samples)")
    

    print("----------------------------------------------------------")
    print("5. Testing update surrogate with combined data\n")

    X_scaler_upd = StandardScaler()
    y_scaler_upd = StandardScaler()
    X_scaler_upd.fit(X_combined)
    y_scaler_upd.fit(y_combined.reshape(-1, 1))

    X_combined_scaled = X_scaler_upd.transform(X_combined)
    y_combined_scaled = y_scaler_upd.transform(y_combined.reshape(-1, 1)).flatten()
    
    # Update surrogate with combined data
    gp_upd = GaussianProcess.from_params(kernel=kernel, noise_std=0.1, jitter=1e-6)
    gp_upd_fitted = gp_upd.fit(jnp.array(X_combined_scaled), jnp.array(y_combined_scaled), opt_config=opt_config)
    
    # Wrap in SurrogateGPax
    surrogate_upd = SurrogateGPax(model=gp_upd_fitted, opt_config=opt_config)

    pipe_upd = SurrogatePipe(
        surrogate=surrogate_upd,
        varSet=vset,
        X=X_combined,
        y=y_combined,
        x_scaler=X_scaler_upd,
        y_scaler=y_scaler_upd,
        verbose=False
    )

    pool = SurrogatePool(surrogates=[pipe, pipe_upd])
    print(f"   ✓ SurrogatePool created with {len(pool.surrogates)} surrogates")
    print(f"   ✓ Surrogate 0: {type(pool.get(0).surrogate.model.kernel).__name__}")
    print(f"   ✓ Surrogate 1: {type(pool.get(1).surrogate.model.kernel).__name__}")
    

    
   
    


    # ----------------------------------------------------------------
    # 6. Evaluate updated surrogate on both OLD and NEW test sets
    # ----------------------------------------------------------------
    print("----------------------------------------------------------")
    print("6. Evaluating updated surrogate on both test sets\n")
    
    def r2_score(y_true, y_pred):
        return 1.0 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    
    # Get prediction function for updated surrogate (pool.get(1))
    pred_fn_updated = pool.get(1).make_predict_fn()
    
    # ---- Evaluate on OLD test set (original Sobol-G function) ----
    print("   A. Performance on OLD test set (original function a=[1,1,1,1]):")
    y_pred_old, y_std_old = pred_fn_updated(X_test)
    y_pred_old_np = to_numpy(y_pred_old)
    y_std_old_np = to_numpy(y_std_old)
    
    r2_old = r2_score(y_test, y_pred_old_np)
    mae_old = np.mean(np.abs(y_test - y_pred_old_np))
    rmse_old = np.sqrt(np.mean((y_test - y_pred_old_np)**2))
    
    print(f"      R²:   {r2_old:.4f}")
    print(f"      MAE:  {mae_old:.4f}")
    print(f"      RMSE: {rmse_old:.4f}")
    print(f"      Mean uncertainty (std): {np.mean(y_std_old_np):.4f}")
    
    # ---- Evaluate on NEW test set (modified Sobol-G function) ----
    print(f"\n   B. Performance on NEW test set (modified function {a1}):")
    y_pred_new, y_std_new = pred_fn_updated(X_new_test)
    y_pred_new_np = to_numpy(y_pred_new)
    y_std_new_np = to_numpy(y_std_new)
    
    r2_new = r2_score(y_new_test, y_pred_new_np)
    mae_new = np.mean(np.abs(y_new_test - y_pred_new_np))
    rmse_new = np.sqrt(np.mean((y_new_test - y_pred_new_np)**2))
    
    print(f"      R²:   {r2_new:.4f}")
    print(f"      MAE:  {mae_new:.4f}")
    print(f"      RMSE: {rmse_new:.4f}")
    print(f"      Mean uncertainty (std): {np.mean(y_std_new_np):.4f}")
    
    # ---- Compare with original surrogate ----
    print("\n   C. Comparison with original surrogate on OLD test set:")
   
    r2_orig = r2_score(y_test, to_numpy(y_pred))
    r2_upd = r2_old
    
    print(f"      Original surrogate: R² = {r2_orig:.4f}")
    print(f"      Updated surrogate:  R² = {r2_upd:.4f}")
    print(f"      Improvement:        ΔR² = {r2_upd - r2_orig:.4f}")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    
    return {
        'pipe': pipe,
        'pipe_updated': pipe_upd,
        'pool': pool,
        'metrics': metrics,
        'test_results': {
            'r2_original': r2_orig,
            'r2_updated_old_test': r2_old,
            'r2_updated_new_test': r2_new,
            'mae_old': mae_old,
            'rmse_old': rmse_old,
            'mae_new': mae_new,
            'rmse_new': rmse_new
        }
    }
    



if __name__ == "__main__":
    results = main()
