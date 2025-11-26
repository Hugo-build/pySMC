"""
The academical research project --> upd surrogates with sobol G
functions

1. making SurrogatePipe after sampling from sobol G function
2. change the parameters of sobol G function for new samples
3. fit new surrogate pipe with 
   - size oriented weighted data
   - novelty oriented weighted data
   - size and novelty oriented weighted data

"""

# %% ########################################################################################
# Import libraries

import sys
from pathlib import Path
# Add parent directory to path for imports when running from examples/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import fields

import numpy as np
import jax.numpy as jnp
from pprint import pprint

from core.Surrogates import (
    SurrogatePipe, 
    StandardScaler, 
    to_numpy,
    calc_upd_weight,
    combine_weighted_data
)

from core.Weighted import SizeNoveltyWeight, SizeWeight, NoveltyWeight
from core.GPax import GaussianProcess, optSetup, RBF
from core.DoEs import sobol_g
from core.Variables import VariableSet, Variable
from core.DataWash import train_test_split
from core.Samplers import sample_inputs

def print_metrics(y_true, y_pred):
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

    # 1. making SurrogatePipe after sampling from sobol G function
    # 2. change the parameters of sobol G function for new samples
    # 3. fit new surrogate pipe with size oriented weighted data
    # 4. fit new surrogate pipe with novelty oriented weighted data

# %% ########################################################################################
# 1. making SurrogatePipe after sampling from sobol G function
# ########################################################################################

# Sobol G-function parameters (controls importance of each dimension)
a = np.array([1.0, 1.0, 1.0])  # Different importance for 4D problem
n_dim = a.size
fn_eval = sobol_g(a)

print(f"\nRunning Sobol G-function with a = {a}")
print(f"Problem dimensionality: {n_dim}")


# Set up the variable set
vset = VariableSet([
    Variable(name=f"x_{i+1}",params ={"low": 0.0, "high": 1.0}) 
    for i in range(n_dim)
])

X = sample_inputs(vset, 300, kind="lhs", seed=42)
y = np.array([fn_eval(x)["y"] for x in X])

print("================================================")
print("Data generated from Sobol G-function")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print("================================================")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
print("================================================")
print("Data split into train and test sets")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}\n")

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print("================================================")


# -------------------------------------------------------
# Set up the parametric GPR model
kernel = RBF.from_params(
    signal_std=float(jnp.std(y_train)),
    length_scale=jnp.ones(n_dim) * 0.2
)
gp = GaussianProcess.from_params(
    kernel=kernel, 
    noise_std=0.1, 
    jitter=1e-5)

opt = optSetup(
    optimizer='adam', 
    steps=300, 
    lr=0.05, 
    verbose=True, 
    log_every=50
)

# Fit the GPR model
gp_fitted = gp.fit(
    jnp.array(X_train), 
    jnp.array(y_train), 
    opt_config=opt)

pipe0 = SurrogatePipe(
    model=gp_fitted,
    varSet=vset,
    X=X_train,
    y=y_train
) # The training data for this pipe is stored in this dataclass

predict_fn = pipe0.make_predict_fn()
y_pred, y_std = predict_fn(X_test)

print(f"   ✓ SurrogatePipe created")
print(f"   ✓ Scaling enabled: X={pipe0._scaled4X}, y={pipe0._scaled4y}")
print(f"   ✓ Model type: {type(pipe0.model).__name__}")
print(f"   ✓ Training data: {pipe0.X.shape[0]} samples, {pipe0.X.shape[1]} dimensions")


y_pred, y_std = predict_fn(X_test)
# Convert to numpy for metrics
y_pred_np = to_numpy(y_pred)
y_std_np = to_numpy(y_std)


print("===============================================================")
print("\n   Prediction metrics:")
metrics = print_metrics(y_test, y_pred_np)
print(f"\n   Mean prediction uncertainty: {np.mean(y_std_np):.4f}")
print(f"   Max prediction uncertainty:  {np.max(y_std_np):.4f}")
print(f"   Min prediction uncertainty:  {np.min(y_std_np):.4f}")
print("===============================================================")

print("\n   ✓ SurrogatePipe created")
pprint(pipe0.__annotations__)    








# %% ########################################################################################
# 2. change the parameters of sobol G function for new samples
# ########################################################################################

# Change the parameters of the sobol G function
a = np.array([0.5, 1.0, 2.0])  # Different importance for 4D problem
n_dim = a.size
fn_eval = sobol_g(a)

X = sample_inputs(vset, 300, kind="lhs", seed=42)
y = np.array([fn_eval(x)["y"] for x in X])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
print("================================================")
print("Data split into train and test sets")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}\n")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print("================================================")

print("================================================")
print(" Old data")
print(f"X shape: {pipe0.X.shape}")
print(f"y shape: {pipe0.y.shape}")

print(" \nNew data:")
print(f"X shape: {X_train.shape}")
print(f"y shape: {y_train.shape}")
print("================================================")

weight_SSNS = calc_upd_weight(
    X_old = pipe0.X,
    y_old = pipe0.y,
    X_new = X_train,
    y_new = y_train,
    predict_fn = predict_fn,
    strategy = SizeNoveltyWeight(novelty_power=0.5),
    verbose = True
)

print(f"   ✓ Calculated adaptive weight: {weight_SSNS:.4f}")

X_combined, y_combined = combine_weighted_data(
    X_old = pipe0.X,
    y_old = pipe0.y,
    X_new = X_train,
    y_new = y_train,
    weight = weight_SSNS,
    verbose = True
)

print(f"   ✓ Combined dataset: {X_combined.shape[0]} samples")
print(f"     (from {pipe0.X.shape[0]} old + {X_train.shape[0]} new samples)")


# %% ########################################################################################
