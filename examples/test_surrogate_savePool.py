

"""
Test the Surrogate framework with save/load functionality.

This test demonstrates:
1. Creating GP models and SurrogatePipe with scalers
2. Save/load for StandardScaler
3. Save/load for SurrogatePipe (JSON and folder modes)
4. Save/load for SurrogatePool
5. Verification that loaded models produce same predictions

Reference: examples/test_surrogate_pipe_simple.py
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field 
from typing import Protocol, Dict, List, Any, Tuple, Optional, Union, Callable, Literal
from enum import Enum
import json

import sys
from pathlib import Path
# Add parent directory to path for imports when running from examples/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))



import jax.numpy as jnp
import numpy as np
Array = np.ndarray

import shutil
from pprint import pprint

# Import GPax components
from core.GPax import GaussianProcess, RBF, optSetup
from core.Surrogates import SurrogatePipe, SurrogatePool, StandardScaler, to_numpy, SurrogateGPax
from core.DoEs import sobol_g
from core.Variables import VariableSet, Variable
from core.DataWash import train_test_split
from core.Samplers import sample_inputs

def print_section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def print_metrics(y_true, y_pred):
    """Calculate and print prediction metrics."""
    r2 = 1.0 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return {'R²': float(r2), 'MAE': float(mae), 'RMSE': float(rmse)}

# Create a real folder for saving test outputs (in savings/ directory)
test_dir = Path(__file__).parent.parent / "savings" / "surrogate_pool_test"
test_dir.mkdir(parents=True, exist_ok=True)
print(f"Test directory: {test_dir}")

try:
    # ===========================================================
    print_section("1. Setup: Sobol G-function test problem")
    # ===========================================================
    
    a = np.array([0.5, 1.0, 2.0])
    n_dim = a.size
    f = sobol_g(a)
    
    print(f"Sobol G-function with a = {a}")
    print(f"Dimensionality: {n_dim}")
    
    vset = VariableSet([
        Variable(name=f"x_{i+1}", kind="uniform", params={"low": 0.0, "high": 1.0})
        for i in range(n_dim)
    ])
    
    # Generate data
    X = sample_inputs(vset, 200, kind="lhs", seed=42)
    y = np.array([f(x)["y"] for x in X])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    
    print(f"Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"y range: [{y.min():.4f}, {y.max():.4f}]")
    
    # ===========================================================
    print_section("2. Test StandardScaler save/load")
    # ===========================================================
    
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    
    print(f"X scaler mean: {x_scaler.mean_}")
    print(f"Y scaler mean: {y_scaler.mean_[0]:.4f}")
    
    # Save scalers
    x_scaler.save(test_dir / "x_scaler.json")
    y_scaler.save(test_dir / "y_scaler.json")
    
    # Load scalers
    x_scaler_loaded = StandardScaler.load(test_dir / "x_scaler.json")
    y_scaler_loaded = StandardScaler.load(test_dir / "y_scaler.json")
    
    # Verify
    assert np.allclose(x_scaler.mean_, x_scaler_loaded.mean_), "X scaler mean mismatch!"
    assert np.allclose(y_scaler.scale_, y_scaler_loaded.scale_), "Y scaler scale mismatch!"
    print("\n✓ StandardScaler save/load verified!")
    
    # ===========================================================
    print_section("3. Create and fit GP model")
    # ===========================================================
    
    # Scale data for training
    X_train_scaled = x_scaler.transform(X_train)
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).flatten()
    
    # Create GP
    kernel = RBF.from_params(
        signal_std=float(np.std(y_train_scaled)),
        length_scale=jnp.ones(n_dim) * 0.3
    )
    gp = GaussianProcess.from_params(kernel=kernel, noise_std=0.05)
    
    # Fit with optimization
    opt_config = optSetup(
        optimizer='adam',
        steps=100,
        lr=0.02,
        verbose=False,
        log_every=50
    )
    
    gp_fitted = gp.fit(
        jnp.array(X_train_scaled),
        jnp.array(y_train_scaled),
        opt_config=opt_config
    )
    print(f"GP fitted with {X_train_scaled.shape[0]} samples")
    
    # ===========================================================
    print_section("4. Create SurrogatePipe with scalers")
    # ===========================================================
    
    pipe = SurrogatePipe(
        surrogate=gp_fitted,
        varSet=vset,
        X=X_train,
        y=y_train,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        verbose=True
    )
    
    print(f"✓ SurrogatePipe created")
    print(f"  - Scaling: X={pipe._scaled4X}, y={pipe._scaled4y}")
    print(f"  - Model type: {pipe._detect_model_type()}")
    
    # Make predictions with original pipe
    predict_fn = pipe.make_predict_fn()
    y_pred_orig, y_std_orig = predict_fn(X_test)
    y_pred_orig = to_numpy(y_pred_orig)
    
    metrics_orig = print_metrics(y_test, y_pred_orig)
    print(f"\nOriginal pipe metrics: R²={metrics_orig['R²']:.4f}, MAE={metrics_orig['MAE']:.4f}")
    
    # ===========================================================
    print_section("5. Test SurrogatePipe save/load (JSON mode)")
    # ===========================================================
    
    # Save as JSON
    pipe.save(test_dir / "pipe.json", as_folder=False, include_data=True)
    
    # Load from JSON
    pipe_loaded_json = SurrogatePipe.load(test_dir / "pipe.json")
    
    # Verify predictions match
    predict_fn_json = pipe_loaded_json.make_predict_fn()
    y_pred_json, _ = predict_fn_json(X_test)
    y_pred_json = to_numpy(y_pred_json)
    
    pred_diff_json = np.max(np.abs(y_pred_orig - y_pred_json))
    print(f"Max prediction difference (JSON): {pred_diff_json:.2e}")
    assert pred_diff_json < 1e-5, "JSON load predictions don't match!"
    print("✓ SurrogatePipe JSON save/load verified!")
    
    # ===========================================================
    print_section("6. Test SurrogatePipe save/load (folder mode)")
    # ===========================================================
    
    # Save as folder
    pipe.save(test_dir / "pipe_folder", as_folder=True, include_data=True)
    
    # Load from folder
    pipe_loaded_folder = SurrogatePipe.load(test_dir / "pipe_folder")
    
    # Verify predictions match
    predict_fn_folder = pipe_loaded_folder.make_predict_fn()
    y_pred_folder, _ = predict_fn_folder(X_test)
    y_pred_folder = to_numpy(y_pred_folder)
    
    pred_diff_folder = np.max(np.abs(y_pred_orig - y_pred_folder))
    print(f"Max prediction difference (folder): {pred_diff_folder:.2e}")
    assert pred_diff_folder < 1e-5, "Folder load predictions don't match!"
    print("✓ SurrogatePipe folder save/load verified!")
    
    # ===========================================================
    print_section("7. Test SurrogatePool save/load")
    # ===========================================================
    
    # Create a second pipe with different kernel
    kernel2 = RBF.from_params(
        signal_std=float(np.std(y_train_scaled)),
        length_scale=jnp.ones(n_dim) * 0.5
    )
    gp2 = GaussianProcess.from_params(kernel=kernel2, noise_std=0.1)
    gp2_fitted = gp2.fit(
        jnp.array(X_train_scaled),
        jnp.array(y_train_scaled),
        opt_config=opt_config
    )
    
    # Wrap in SurrogateGPax
    surrogate2 = SurrogateGPax(model=gp2_fitted, opt_config=opt_config)

    pipe2 = SurrogatePipe(
        surrogate=surrogate2,
        X=X_train,
        y=y_train,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )
    
    # Create pool
    pool = SurrogatePool([pipe, pipe2])
    print(f"Created pool with {len(pool)} pipes")
    
    # Save pool
    pool.save(test_dir / "pool", include_data=True)
    
    # Load pool
    pool_loaded = SurrogatePool.load(test_dir / "pool")
    print(f"Loaded pool with {len(pool_loaded)} pipes")
    
    # Verify predictions from first pipe in pool
    predict_fn_pool = pool_loaded[0].make_predict_fn()
    y_pred_pool, _ = predict_fn_pool(X_test)
    y_pred_pool = to_numpy(y_pred_pool)
    
    pred_diff_pool = np.max(np.abs(y_pred_orig - y_pred_pool))
    print(f"Max prediction difference (pool[0]): {pred_diff_pool:.2e}")
    assert pred_diff_pool < 1e-5, "Pool load predictions don't match!"
    print("✓ SurrogatePool save/load verified!")
    
    # Print pool summary
    print("\nPool summary:")
    pprint(pool_loaded.to_summary())
    
    # ===========================================================
    print_section("8. Summary")
    # ===========================================================
    
    print("All tests passed!")
    print(f"✓ StandardScaler save/load")
    print(f"✓ SurrogatePipe save/load (JSON)")
    print(f"✓ SurrogatePipe save/load (folder)")
    print(f"✓ SurrogatePool save/load")
    print(f"\nOriginal model R²: {metrics_orig['R²']:.4f}")
    print(f"Test directory: {test_dir}")
    
finally:
    # To clean up saved files, uncomment the following line:
    # shutil.rmtree(test_dir)
    print(f"\n[Saved files location: {test_dir}]")
    print("="*70)

