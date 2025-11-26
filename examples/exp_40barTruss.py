# %% ########################################################################################
# Import libraries
import sys
from pathlib import Path
# Add parent directory to path for imports when running from examples/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import norm
from SALib.analyze.sobol import analyze as SA_analyze
from typing import Tuple, Dict, Callable, Optional
from pprint import pprint

from lib.FElib import elMatrixBar6DoF, rotateMat
from core.GPax import RBF, GaussianProcess, optSetup
from core.DataWash import train_test_split, scale_data

# Set global font size
plt.rcParams.update({'font.size': 18})  # Adjust the number as needed

# Or more specific control:
plt.rcParams.update({
    'font.size': 12,           # Default font size
    'axes.titlesize': 18,      # Title font size
    'axes.labelsize': 18,      # Axis label font size
    'xtick.labelsize': 18,     # X-axis tick label size
    'ytick.labelsize': 18,     # Y-axis tick label size
    'legend.fontsize': 18,     # Legend font size
    'figure.titlesize': 18     # Figure title font size
})

# %% #####################################################################################

# Read in the nodes
# ------------------------------------------------
# 1. Node coordinates: node_id -> [x, y, z] (mm)
# ------------------------------------------------
file_folder = '40barTruss'
nodes_df = pd.read_csv(f'{file_folder}/nodes.csv')

# Convert nodes DataFrame to a list of dictionaries with 'id' field
nodes = [
    {
        'id': int(row['node_id']),
        'coords': np.array([row['x'], row['y'], row['z']])
    }
    for _, row in nodes_df.iterrows()
]
print("Nodes loaded:", len(nodes))
print(nodes_df.to_string(index=False)) 

# --------------------------------------------------------
# 2. Element connectivity: [node_i, node_j] + properties
# --------------------------------------------------------
elements_df = pd.read_csv(f'{file_folder}/elements.csv')
# Convert elements DataFrame to a list of dictionaries with 'id' field
elements = [
    {
        'id': int(row['element_id']),
        'node_i': int(row['node_i']),
        'node_j': int(row['node_j']),
        'E': row['E'],
        'A': row['A'],
        'rho': row['rho']
    }
    for _, row in elements_df.iterrows()
]
print("Elements loaded:", len(elements))
print(elements_df.to_string(index=False)) 

# -------------------------------
# plot the structure
# -------------------------------
# Create a lookup dictionary for nodes by id for efficient access
nodes_by_id = {node['id']: node for node in nodes}

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for elem in elements:
    ni = elem['node_i']
    nj = elem['node_j']
    node_i = nodes_by_id[ni]
    node_j = nodes_by_id[nj]
    p1 = node_i['coords']
    p2 = node_j['coords']
    ax.plot(*zip(p1, p2), color='b', linewidth=1)

    # add node numbers
    ax.text(p1[0], p1[1], p1[2], str('N'+str(ni)), color='r')
    ax.text(p2[0], p2[1], p2[2], str('N'+str(nj)), color='r')

plt.show()


F = np.zeros(3 * len(nodes))
F[3*(10-1) + 1] = 1000  # 1000 N right at node 10
FE_config = {'nodes': nodes, 
         'elements': elements,
         'fixed_dofs': [0,1,2, 3,4,5, 6,7,8, 9,10,11],
         'loads': F}

pprint(FE_config)





# %% #####################################################################################
# main function to solve the FE problem
def solve_FE_static(FE_config):
    nodes = FE_config['nodes']
    elements = FE_config['elements']
    fixed_dofs = FE_config['fixed_dofs']
    loads = FE_config['loads']

    # Create a lookup dictionary for nodes by id for efficient access
    nodes_by_id = {node['id']: node for node in nodes}

    K = np.zeros((3 * len(nodes), 3 * len(nodes)))
    for elem in elements:
        
        ni = elem['node_i']
        nj = elem['node_j']
        node_i = nodes_by_id[ni]
        node_j = nodes_by_id[nj]
        xi = np.array(node_i['coords'])
        xj = np.array(node_j['coords'])
        L = np.linalg.norm(xj - xi)
        
        # Create temporary node objects for rotation matrix calculation
        node1 = xi
        node2 = xj

        # Get rotation-transformation matrix
        T = rotateMat(node1, node2)
        
        T_full = np.zeros((6,6))
        T_full[0:3, 0:3] = T  # for node i
        T_full[3:6, 3:6] = T  # for node j

        # Get element stiffness matrix in local coordinates
        k_local, _ = elMatrixBar6DoF(E=elem['E'], A=elem['A'], rho=elem['rho'], L=L)

        # Transform element stiffness to global coordinates
        k = T_full.T @ k_local @ T_full

        # DoF indices
        dof_i = [3*(ni-1), 3*(ni-1)+1, 3*(ni-1)+2]
        dof_j = [3*(nj-1), 3*(nj-1)+1, 3*(nj-1)+2]
        dofs = dof_i + dof_j
       
        # Assmeble local to global
        K[np.ix_(dof_i, dof_i)] += k[0:3, 0:3]
        K[np.ix_(dof_j, dof_j)] += k[3:6, 3:6]
        K[np.ix_(dof_i, dof_j)] += k[0:3, 3:6]
        K[np.ix_(dof_j, dof_i)] += k[3:6, 0:3]

    # Apply boundary conditions ( eliminate fixed DoFs)
    all_dofs = np.arange(3 * len(nodes))
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f = loads[free_dofs]

    # Solve for displacements
    U = np.zeros(3 * len(nodes))
    U[free_dofs] = np.linalg.solve(K_ff, F_f)

    return U


# %% #####################################################################################
# Example solving of the FE problem
U = solve_FE_static(FE_config)
print(U)


def plot_displacements(U, FE_config, scale=1000.0):
    nodes = FE_config['nodes']
    elements = FE_config['elements']

    # Create a lookup dictionary for nodes by id for efficient access
    nodes_by_id = {node['id']: node for node in nodes}
    
    # Plot the displacements
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate the bounds of the structure for equal aspect ratio
    x_coords = []
    y_coords = []
    z_coords = []
    
    for elem in elements:
        ni = elem['node_i']
        nj = elem['node_j']
        node_i = nodes_by_id[ni]
        node_j = nodes_by_id[nj]
        x_coords.extend([node_i['coords'][0], node_j['coords'][0]])
        y_coords.extend([node_i['coords'][1], node_j['coords'][1]])
        z_coords.extend([node_i['coords'][2], node_j['coords'][2]])

    # Calculate max displacement for display
    max_displacement = max(abs(U))
    
    for i, elem in enumerate(elements):
        ni = elem['node_i']
        nj = elem['node_j']
        node_i = nodes_by_id[ni]
        node_j = nodes_by_id[nj]
        p1 = node_i['coords']
        p2 = node_j['coords']
        
        # Plot original structure (only add label for first element to avoid legend clutter)
        if i == 0:
            ax.plot(*zip(p1, p2), color='b', linewidth=1, label='Undeformed')
        else:
            ax.plot(*zip(p1, p2), color='b', linewidth=1)
        
        # Plot deformed structure
        d1 = p1 + scale * U[3*(ni-1):3*(ni-1)+3]
        d2 = p2 + scale * U[3*(nj-1):3*(nj-1)+3]
        if i == 0:
            ax.plot(*zip(d1, d2), color='r', linewidth=2, label='Deformed')
        else:
            ax.plot(*zip(d1, d2), color='r', linewidth=2)

    # Set title with max displacement info
    title = f'Original vs Deformed Structure\nDeformation scale: {scale}x (Max displacement: {max_displacement:.2e} mm)'
    ax.set_title(title)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.legend()
    
    # Make the plot more readable
    ax.grid(True)
    
    # Set equal aspect ratio
    max_range = np.array([max(x_coords) - min(x_coords),
                         max(y_coords) - min(y_coords),
                         max(z_coords) - min(z_coords)]).max() / 2.0
    mid_x = (max(x_coords) + min(x_coords)) * 0.5
    mid_y = (max(y_coords) + min(y_coords)) * 0.5
    mid_z = (max(z_coords) + min(z_coords)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    

    return fig


plot_displacements(U, FE_config, scale=50.0)



"""
SUMMARY

- The FE analysis is performed using the FElib library.
- ThE MAIN FUNCTION IS solve_FE_static(FE_config)
- The "FE_config" is a dictionary that contains the nodes, elements, fixed dofs, and loads.

- The "nodes" is a list of dictionaries with the node id and coordinates.
- The "elements" is a list of dictionaries with the element id, node i, node j, E, A, and rho.
- The "fixed_dofs" is a list of the fixed dofs.
- The "loads" is a list of the loads.

- The "U" is the displacement vector, returned by the solve_FE_static function.
"""





# %% #####################################################################################
# Parametric Study using Variable.targets feature
# ########################################################################################

from core.Variables import Variable, VariableSet, inject_single_config

print("\n" + "="*70)
print("PARAMETRIC STUDY: Using Variable.targets for config injection")
print("="*70)

# Define variables with targets (paths to config locations)
# -------------------------------------------------------------------------------

# Variable 1: Young's modulus - apply to all elements using wildcard
E_var = Variable(
    name="E",
    kind="uniform",
    params={"low": 200e7, "high": 220e7}
)
E_var.add_target(doc="FE", path="elements[*].E")

# Variable 2: Load magnitude at node 10 (y-direction)
# Index = 3*(10-1) + 1 = 29
load_var = Variable(
    name="F_y",
    kind="uniform",
    params={"low": 500, "high": 1500}
)
load_var.add_target(doc="FE", path="loads[29]")

# Variable 3: Cross-sectional area for bottom chord elements (first 8 elements)
A_bottom = Variable(
    name="A_bottom",
    kind="uniform",
    params={"low": 90, "high": 110}
)
# Add multiple targets for multiple elements
for i in range(8):
    A_bottom.add_target(doc="FE", path=f"elements[{i}].A")

# Variable 4: Material density - apply to all
rho_var = Variable(
    name="rho",
    kind="normal",
    params={"mean": 7.85e-6, "std": 0.1e-6}
)
rho_var.add_target(doc="FE", path="elements[*].rho")

# Create variable set
variables_list = [E_var, load_var, A_bottom, rho_var]
var_set = VariableSet(variables=variables_list)

print(f"\nDefined {len(variables_list)} variables with targets:")
for var in variables_list:
    print(f"  • {var.name}: {len(var.targets)} target(s)")
    for target in var.targets[:2]:  # Show first 2 targets
        print(f"    --> {target['path']}")
    if len(var.targets) > 2:
        print(f"    ... and {len(var.targets)-2} more")


pprint(var_set.to_SAlib())



# %% #####################################################################################
# TEST 1: Manual injection of specific values
# ########################################################################################

print("\n" + "-"*70)
print("METHOD 1: Manual value injection")
print("-"*70)

# Prepare configs dict with document name
configs = {"FE": FE_config}

# Manually specify values
manual_values = {
    "E": 210e3,
    "F_y": 1200,
    "A_bottom": 95,
    "rho": 7.9e-6
}

# Inject values
modified_configs = var_set.inject_values(configs, manual_values)
modified_FE = modified_configs["FE"]

# Verify changes
print(f"\nOriginal config - Element 0 E: {FE_config['elements'][0]['E']:.2e} MPa")
print(f"Modified config - Element 0 E: {modified_FE['elements'][0]['E']:.2e} MPa")
print(f"\nOriginal config - Element 0 A: {FE_config['elements'][0]['A']:.2f} mm²")
print(f"Modified config - Element 0 A: {modified_FE['elements'][0]['A']:.2f} mm²")
print(f"\nOriginal config - Load[29]: {FE_config['loads'][29]:.1f} N")
print(f"Modified config - Load[29]: {modified_FE['loads'][29]:.1f} N")

# Solve modified config
U_modified = solve_FE_static(modified_FE)
print(f"\nMax displacement (original): {np.max(np.abs(U)):.6e} mm")
print(f"Max displacement (modified): {np.max(np.abs(U_modified)):.6e} mm")

# %% #####################################################################################
# TEST 2: Automatic sampling for parametric study
# ########################################################################################

print("\n" + "-"*70)
print("METHOD 2: Automatic sampling")
print("-"*70)

# Generate samples - use just 2 variables for clearer visualization
vars_simple = [E_var, load_var]
var_set_simple = VariableSet(variables=vars_simple)

rng = np.random.default_rng(seed=42)
n_samples = (2*len(vars_simple)+2)*100

# Sample configurations
sampled_configs, samples = var_set_simple.sample_configs(
    configs={"FE": FE_config},
    n_samples=n_samples,
    rng=rng
)

print(f"\nGenerated {len(sampled_configs)} configurations")
print(f"Sample array shape: {samples.shape}")
print("\nFirst 5 samples:")
print(f"{'Sample':<8} {'E (MPa)':<15} {'F_y (N)':<12}")
print("-" * 40)
for i in range(min(10, n_samples)):
    print(f"{i:<8} {samples[i,0]:<15.2f} {samples[i,1]:<12.1f}")

# %% #####################################################################################
# Test 3: Run parametric study and analyze
# ########################################################################################

print("\n" + "-"*70)
print("METHOD 3: Solve all configurations")
print("-"*70)

# Solve each configuration
results = []
for i, config_dict in enumerate(sampled_configs):
    fe_config = config_dict["FE"]
    U_sample = solve_FE_static(fe_config)
    max_disp = np.max(np.abs(U_sample))
    
    results.append({
        'id': i,
        'E': samples[i, 0],
        'F_y': samples[i, 1],
        'max_displacement': max_disp
    })

print(f"\nSolved {len(results)} FE problems")
print(f"  Min max displacement: {min(r['max_displacement'] for r in results):.6e} mm")
print(f"  Max max displacement: {max(r['max_displacement'] for r in results):.6e} mm")
print(f"  Mean max displacement: {np.mean([r['max_displacement'] for r in results]):.6e} mm")

# %% #####################################################################################
# Visualization: Parameter sensitivity
# ########################################################################################

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: E vs Displacement
ax1 = axes[0]
scatter1 = ax1.scatter(
    [r['E']/1e3 for r in results],  # Convert to GPa for readability
    [r['max_displacement'] for r in results],
    c=[r['F_y'] for r in results],
    s=100,
    cmap='plasma',
    edgecolor='black'
)
ax1.set_xlabel('Young\'s Modulus [GPa]')
ax1.set_ylabel('Max Displacement [mm]')
ax1.set_title('Stiffness vs Displacement')
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Load [N]')
ax1.grid(True, alpha=0.3)

# Plot 2: Load vs Displacement
ax2 = axes[1]
scatter2 = ax2.scatter(
    [r['F_y'] for r in results],
    [r['max_displacement'] for r in results],
    c=[r['E']/1e3 for r in results],
    s=100,
    cmap='viridis',
    edgecolor='black'
)
ax2.set_xlabel('Load [N]')
ax2.set_ylabel('Max Displacement [mm]')
ax2.set_title('Load vs Displacement')
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('E [GPa]')
ax2.grid(True, alpha=0.3)

# Plot 3: 2D parameter space
ax3 = axes[2]
scatter3 = ax3.scatter(
    [r['E']/1e3 for r in results],
    [r['F_y'] for r in results],
    c=[r['max_displacement'] for r in results],
    s=150,
    cmap='coolwarm',
    edgecolor='black'
)
ax3.set_xlabel('Young\'s Modulus [GPa]')
ax3.set_ylabel('Load [N]')
ax3.set_title('Parameter Space')
cbar3 = plt.colorbar(scatter3, ax=ax3)
cbar3.set_label('Max Displacement [mm]')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
#plt.savefig('figs/parametric_study_40barTruss.png', dpi=150, bbox_inches='tight')
print("\n✓ Plots saved to: figs/parametric_study_40barTruss.png")
plt.show()



# %% #####################################################################################
# Test 4: SAlib workflow
# ########################################################################################

print("\n" + "-"*70)
print("METHOD 4: SAlib workflow")
print("-"*70)

# Prepare configs dict with document name
configs = {"FE": FE_config}

SA_prob = var_set_simple.to_SAlib()
pprint(SA_prob)

# attribute the input samples
X = samples
y = np.array([r['max_displacement'] for r in results])

SA_results = SA_analyze(SA_prob, y, print_to_console=True)





# %% #####################################################################################
# Test 5: prepare data for surrogate model
# ########################################################################################
import jax.numpy as jnp

print("\n" + "-"*70)
print("METHOD 5: train a surrogate model")
print("-"*70)

vars = [E_var, load_var, A_bottom, rho_var]
var_set = VariableSet(variables=vars)

samples = var_set.sample_configs(configs={"FE": FE_config}, n_samples=n_samples, rng=rng)
rng = np.random.default_rng(seed=42)
n_samples = (2*len(vars_simple)+2)*100

# Sample configurations
sampled_configs, samples = var_set.sample_configs(
    configs={"FE": FE_config},
    n_samples=n_samples,
    rng=rng
)

print(f"\nGenerated {len(sampled_configs)} configurations")
print(f"Sample array shape: {samples.shape}")
print("\nFirst 5 samples:")
print(f"{'Sample':<8} {'E (MPa)':<15} {'F_y (N)':<12}")
print("-" * 40)
for i in range(min(10, n_samples)):
    print(f"{i:<8} {samples[i,0]:<15.2f} {samples[i,1]:<12.1f}")

X = samples
# ------------------------------------------------------------
# Solve each configuration
results = []
for i, config_dict in enumerate(sampled_configs):
    fe_config = config_dict["FE"]
    U_sample = solve_FE_static(fe_config)
    max_disp = np.max(np.abs(U_sample))
    
    results.append({
        'id': i,
        'E': samples[i, 0],
        'F_y': samples[i, 1],
        'A_bottom': samples[i, 2],
        'rho': samples[i, 3],
        'max_displacement': max_disp
    })

y = np.array([r['max_displacement'] for r in results])





# %% #####################################################################################
# Test 6: train a surrogate model WITH SCALING
# ########################################################################################

# Split data first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Scale the data to improve training
print("\n" + "="*70)
print("SCALING DATA FOR IMPROVED TRAINING")
print("="*70)
print("\nOriginal data statistics:")
print(f"X_train: min={X_train.min():.2e}, max={X_train.max():.2e}, mean={X_train.mean():.2e}, std={X_train.std():.2e}")
print(f"y_train: min={y_train.min():.2e}, max={y_train.max():.2e}, mean={y_train.mean():.2e}, std={y_train.std():.2e}")

# Apply scaling
X_train_scaled, y_train_scaled, x_scaler, y_scaler = scale_data(X_train, y_train)
X_test_scaled = x_scaler.transform(X_test)

print("\nScaled data statistics:")
print(f"X_train_scaled: min={X_train_scaled.min():.2e}, max={X_train_scaled.max():.2e}, mean={X_train_scaled.mean():.2e}, std={X_train_scaled.std():.2e}")
print(f"y_train_scaled: min={y_train_scaled.min():.2e}, max={y_train_scaled.max():.2e}, mean={y_train_scaled.mean():.2e}, std={y_train_scaled.std():.2e}")
print("="*70)

opt_config = optSetup(
    optimizer='adam',
    steps=200,
    lr=0.01,
    verbose=True,
    log_every=10
)

# Train a surrogate model on SCALED data
kernel = RBF.from_params(
    signal_std=float(jnp.std(y_train_scaled)),
    length_scale=jnp.ones(X_train_scaled.shape[1]) * 0.1
)
gp = GaussianProcess.from_params(kernel=kernel, noise_std=0.1, jitter=1e-6)
gp_fitted = gp.fit(X_train_scaled, y_train_scaled, opt_config=opt_config)

# Predict using the fitted GP on SCALED test data
y_pred_scaled, y_std_scaled = gp_fitted.predict(X_test_scaled)

# Unscale predictions back to original scale
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_std = y_std_scaled * y_scaler.scale_[0]  # Scale the standard deviation

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


print("\n" + "="*70)
print("MODEL PERFORMANCE ON TEST SET")
print("="*70)
metrics = print_metrics(y_test, y_pred)

# %% #####################################################################################
# Visualization: Compare actual vs predicted
# ########################################################################################

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Predicted vs Actual
ax1 = axes[0]
ax1.scatter(y_test, y_pred, alpha=0.6, s=100, edgecolors='k', linewidths=0.5)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
ax1.set_xlabel('Actual Displacement [mm]')
ax1.set_ylabel('Predicted Displacement [mm]')
ax1.set_title('Predicted vs Actual')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.95, f"R² = {metrics['R²']:.4f}\nRMSE = {metrics['RMSE']:.2e}", 
         transform=ax1.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Plot 2: Prediction with uncertainty
ax2 = axes[1]
indices = np.arange(len(y_test))
ax2.errorbar(indices, y_pred, yerr=2*y_std, fmt='o', capsize=3, alpha=0.6, 
             label='Predicted ± 2σ', markersize=6)
ax2.scatter(indices, y_test, color='red', marker='x', s=100, linewidths=2, 
            label='Actual', zorder=5)
ax2.set_xlabel('Test Sample Index')
ax2.set_ylabel('Displacement [mm]')
ax2.set_title('Predictions with Uncertainty')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('GP Surrogate Model Performance (with Data Scaling)', fontsize=16, fontweight='bold')
plt.tight_layout()
#plt.savefig('figs/surrogate_performance_scaled.png', dpi=150, bbox_inches='tight')
print("\n✓ Performance plots saved to: figs/surrogate_performance_scaled.png")
plt.show()





# %% #####################################################################################
# Test 7: use the surrogate model to calculate SA scores
# ########################################################################################

print("\n" + "-"*70)
print("METHOD 7: calculate SA scores using GP surrogate")
print("-"*70)

SA_prob = var_set.to_SAlib()

# Generate new samples for SA
n_samples_SA = (2*len(vars)+2)*1000
sampled_configs_SA, samples_SA = var_set.sample_configs(
    configs={"FE": FE_config}, 
    n_samples=n_samples_SA, 
    rng=np.random.default_rng(seed=123)
)

print(f"\nGenerated {n_samples_SA} samples for SA")
print(f"Sample array shape: {samples_SA.shape}")

# Use GP surrogate to predict instead of running expensive FE solver
print("\nUsing GP surrogate model to predict displacement...")

# Scale the SA samples using the same scaler from training
X_SA_scaled = x_scaler.transform(samples_SA)

# Predict using the fitted GP model
predict_fn = lambda X: gp_fitted.predict(X)
y_SA_pred_scaled, y_SA_std_scaled = predict_fn(X_SA_scaled)

# Unscale predictions back to original scale
y_SA_pred = y_scaler.inverse_transform(y_SA_pred_scaled.reshape(-1, 1)).ravel()

print(f"\nPredicted displacements statistics:")
print(f"  Min: {y_SA_pred.min():.6e} mm")
print(f"  Max: {y_SA_pred.max():.6e} mm")
print(f"  Mean: {y_SA_pred.mean():.6e} mm")
print(f"  Std: {y_SA_pred.std():.6e} mm")

pprint(SA_prob)

# Calculate SA scores using GP predictions
print("\nCalculating Sobol indices using GP surrogate predictions...")
SA_results = SA_analyze(SA_prob, y_SA_pred, print_to_console=True)




# %% #####################################################################################