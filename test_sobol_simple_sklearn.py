"""
Test Sobol G-function with scikit-learn Gaussian Process.

This demonstrates:
1. Using the Sobol G-function as a test problem
2. Training and testing with sklearn GP
3. Comparison with JAX GP implementation (see test_sobol_simple.py)
"""

import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel

from core.DoEs import sobol_g
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
    plt.plot(idx, y_pred, label="Predicted", color=color_pred, linewidth=2, alpha=0.8)
    plt.scatter(idx, y_true, label="True", color=color_true, s=30, alpha=0.7, zorder=3)
    plt.fill_between(idx, y_pred - 2*y_std, y_pred + 2*y_std, 
                     alpha=0.25, color=color_fill, label="95% confidence", zorder=1)
    plt.xlabel("Sample Index", fontsize=11)
    plt.ylabel("Value", fontsize=11)
    plt.title("Sobol G-Function (sklearn)", fontsize=12, fontweight='bold')
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


# Sobol G-function parameters (controls importance of each dimension)
a = np.array([1.0, 1.0, 1.0, 1.0])  # Equal importance for 4D problem
n_dim = a.size

# Use the variable set to generate the data
vset = VariableSet([
    Variable(name=f"x_{i+1}", kind="uniform", params={"low": 0.0, "high": 1.0})
    for i in range(a.size)
])

f = sobol_g(a)
print(f"Running Sobol G-function with a = {a}")
print(f"Problem dimensionality: {n_dim}")

X = sample_inputs(vset, 500, kind="lhs", seed=42)
y = np.array([f(x)["y"] for x in X])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

print("================================================")
print("Results with sklearn RBF kernel (default optimization):")

# sklearn GP with RBF kernel
# RBF kernel: C(1.0) * RBF(length_scale=1.0)
# WhiteKernel for noise (equivalent to log_sn2 in JAX version)
kernel_rbf = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(n_dim) * 0.1, 
                                        length_scale_bounds=(1e-2, 1e2)) + \
             WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-3, 1e1))

gp_rbf = GaussianProcessRegressor(
    kernel=kernel_rbf,
    n_restarts_optimizer=10,  # Number of restarts for hyperparameter optimization
    random_state=42,
    normalize_y=False  # We handle scaling separately if needed
)

gp_rbf.fit(X_train, y_train)
y_pred_rbf, y_std_rbf = gp_rbf.predict(X_test, return_std=True)

plot_results(y_test, y_pred_rbf, y_std_rbf)
metrics_rbf = print_metrics(y_test, y_pred_rbf, y_std_rbf)
print(f"Optimized kernel: {gp_rbf.kernel_}")
print("================================================")


print("\n================================================")
print("Results with sklearn Matern kernel (default optimization):")

# sklearn GP with Matern kernel
# Matern kernel with nu=2.5 (similar to Matern52 in JAX)
kernel_matern = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(n_dim) * 0.1,
                                              length_scale_bounds=(1e-2, 1e2),
                                              nu=2.5) + \
                WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-3, 1e1))

gp_matern = GaussianProcessRegressor(
    kernel=kernel_matern,
    n_restarts_optimizer=10,
    random_state=42,
    normalize_y=False
)

gp_matern.fit(X_train, y_train)
y_pred_matern, y_std_matern = gp_matern.predict(X_test, return_std=True)

plot_results(y_test, y_pred_matern, y_std_matern)
metrics_matern = print_metrics(y_test, y_pred_matern, y_std_matern)
print(f"Optimized kernel: {gp_matern.kernel_}")
print("================================================")

