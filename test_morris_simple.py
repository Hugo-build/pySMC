"""
Test Morris simple function with pure JAX Gaussian Process (GP.py).

This demonstrates:
1. Using the Morris simple function as a test problem
2. Training and testing with JAX GP
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pprint import pprint

from core.GPax import GaussianProcess, RBF, Matern32, Matern52, optSetup
from core.DoEs import morris_g
from core.Variables import VariableSet, Variable
from core.DataWash import train_test_split
from core.Samplers import sample_inputs



def print_metrics(y_true, y_pred, y_std=None):
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

def plot_results(y_true, y_pred, y_std):
    # Define a professional color palette
    color_true = '#2E86AB'      # Blue for true values
    color_pred = '#A23B72'      # Magenta/purple for predictions
    color_fill = '#F18F01'      # Orange for confidence interval (lighter)
    color_perfect = '#06A77D'   # Teal for perfect fit line
    color_grid = '#E0E0E0'      # Light gray for grid
    
    plt.figure(figsize=(14, 5))
    
    # Left subplot: Time series with predictions
    plt.subplot(1, 2, 1)
    idx = np.arange(len(y_true))
    plt.plot(idx, y_true, label="True", color=color_true, linewidth=2, alpha=0.8)
    plt.scatter(idx, y_pred, label="Predicted", color=color_pred, s=30, alpha=0.7, zorder=3)
    plt.fill_between(idx, y_pred - 2*y_std, y_pred + 2*y_std, 
                     alpha=0.25, color=color_fill, label="95% confidence", zorder=1)
    plt.xlabel("Sample Index", fontsize=11)
    plt.ylabel("Value", fontsize=11)
    plt.title("Morris Simple Function", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, color=color_grid, linestyle='--')
    plt.legend(fontsize=9, framealpha=0.9)
    
    # Right subplot: Predicted vs True scatter
    plt.subplot(1, 2, 2)
    plt.plot(y_true, y_true, label="Perfect fit", color=color_perfect, 
             linewidth=2, linestyle='--', alpha=0.8)
    plt.scatter(y_true, y_pred, label="Predicted", color=color_pred, 
               s=40, alpha=0.6, edgecolors='white', linewidths=0.5, zorder=3)
    plt.xlabel("True Value", fontsize=11)
    plt.ylabel("Predicted Value", fontsize=11)
    plt.title("Predicted vs True (scatter plot)", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, color=color_grid, linestyle='--')
    plt.legend(fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.show()
    return plt


a = np.array([3.0, 1.0, 0.5, 1.0, 0.7])
n_dim = a.size
# use the variable set to generate the data
vset = VariableSet([
    Variable(name=f"x_{i+1}", kind="uniform", params={"low": 0.0, "high": 1.0})
    for i in range(a.size)
])

f = morris_g(a.size)
print(f)

X = sample_inputs(vset, 100, kind="lhs", seed=42)
y = np.array([f(x)["y"] for x in X])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

kernel = RBF(log_sf=jnp.log(jnp.std(y_train) + 1e-6), log_ls=jnp.log(jnp.ones((n_dim,)) * 0.1))
gp = GaussianProcess(kernel=kernel, log_sn2=jnp.log(jnp.array(0.1**2)), jitter=1e-6)

# Capture the returned fitted GP (immutable dataclass pattern)
gp_fitted = gp.fit(jnp.array(X_train), jnp.array(y_train), opt_config=None)



y_pred, y_std = gp_fitted.predict(jnp.array(X_test))
print("================================================")
print("Results without optimization:")
plot_results(y_test, y_pred, y_std)
metrics = print_metrics(y_test, y_pred, y_std)
print("================================================")






# GP with optimized kernelhyperparameters
opt_config = optSetup(
    optimizer='adam',
    steps=300,
    lr=0.01,
    verbose=True,
    log_every=10
)
gp_opt = gp.fit(jnp.array(X_train), jnp.array(y_train), opt_config=opt_config)

# Predict using the fitted GP
y_pred_opt, y_std_opt = gp_opt.predict(jnp.array(X_test))

print("================================================")
print("Results with optimization:")

fig = plot_results(y_test, y_pred_opt, y_std_opt)
metrics = print_metrics(y_test, y_pred_opt, y_std_opt)