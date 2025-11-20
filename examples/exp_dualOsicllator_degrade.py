# %% ####################################################################################
# Import libraries
import sys
from pathlib import Path
# Add parent directory to path for imports when running from examples/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import norm
from typing import Tuple, Dict, Callable, Optional
from pprint import pprint


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

# %% ####################################################################################
# Define parameters

# Dual mass-spring-damper system parameters
m1 = 1.0  # mass of the first oscillator
m2 = 1.0  # mass of the second oscillator
k1 = 0.3  # spring constant of the first oscillator
k2 = 0.2  # spring constant of the second oscillator
k3 = 0.3  # spring constant of the third oscillator
c1 = 0.3  # damping constant of the first oscillator
c2 = 0.3  # damping constant of the second oscillator
c3 = 0.3  # damping constant of the third oscillator

M = np.array([[m1, 0], [0, m2]])
K = np.array([[k1+k2, -k2], [-k2, k2+k3]])
C = np.array([[c1+c2, -c2], [-c2, c2+c3]])

# %% ####################################################################################
# System Analysis Functions

def calculate_frf(frequencies: np.ndarray, M: np.ndarray, K: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Calculate frequency response function for the dual oscillator system
    
    Parameters:
    -----------
    frequencies : array
        Frequency range to calculate FRF (rad/s)
    M, K, C : arrays
        Mass, stiffness, and damping matrices
    
    Returns:
    --------
    H : complex array
        Transfer function matrix
    """
    H = np.zeros((len(frequencies), 2, 2), dtype=complex)
    
    for i, omega in enumerate(frequencies):
        # H(ω) = (-ω²M + iωC + K)^(-1)
        Z = -omega**2 * M + 1j * omega * C + K
        H[i] = np.linalg.inv(Z)
    
    return H


def state_space_form(M: np.ndarray, K: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the state space form of the system
    
    Parameters:
    -----------
    M, K, C : arrays
        Mass, stiffness, and damping matrices
    
    Returns:
    --------
    Arb, Brb : arrays
        State space matrices
    """
    # Construct the state space form of the system
    Arb = np.vstack([np.hstack([np.zeros((2, 2)), np.eye(2)]), 
                     np.hstack([-np.linalg.inv(M) @ K, -np.linalg.inv(M) @ C])])
    Brb = np.vstack([np.zeros((2, 2)), np.linalg.inv(M)])
    
    return Arb, Brb


# %% ####################################################################################
# Calculate eigenfrequencies and mode shapes

# Solve the generalized eigenvalue problem: K*phi = lambda*M*phi
# This gives us eigenvalues (lambda) and eigenvectors (phi - mode shapes)
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(M) @ K)

# Natural frequencies (rad/s)
natural_frequencies_rad = np.sqrt(eigenvalues)

# Natural frequencies (Hz)
natural_frequencies_hz = natural_frequencies_rad / (2 * np.pi)

# Sort by frequency
sort_idx = np.argsort(natural_frequencies_rad)
natural_frequencies_rad = natural_frequencies_rad[sort_idx]
natural_frequencies_hz = natural_frequencies_hz[sort_idx]
mode_shapes = eigenvectors[:, sort_idx]

print('=== EIGENFREQUENCY ANALYSIS ===')
print(f'Mode 1 (rad/s): {natural_frequencies_rad[0]:.4f}')
print(f'Mode 1 (Hz): {natural_frequencies_hz[0]:.4f}')
print(f'Mode 1 shape: [{mode_shapes[0,0]:.4f}, {mode_shapes[1,0]:.4f}]')
print()
print(f'Mode 2 (rad/s): {natural_frequencies_rad[1]:.4f}')
print(f'Mode 2 (Hz): {natural_frequencies_hz[1]:.4f}')
print(f'Mode 2 shape: [{mode_shapes[0,1]:.4f}, {mode_shapes[1,1]:.4f}]')

# Visualize mode shapes
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot([0, 1, 2], [0, mode_shapes[0,0], mode_shapes[1,0]], 'o-', linewidth=2, markersize=8)
plt.title(f'Mode 1: {natural_frequencies_hz[0]:.3f} Hz')
plt.xlabel('Mass Position')
plt.ylabel('Relative Displacement')
plt.grid(True)
plt.xticks([0, 1, 2], ['Fixed', 'Mass 1', 'Mass 2'])

plt.subplot(1, 2, 2)
plt.plot([0, 1, 2], [0, mode_shapes[0,1], mode_shapes[1,1]], 'o-', linewidth=2, markersize=8, color='red')
plt.title(f'Mode 2: {natural_frequencies_hz[1]:.3f} Hz')
plt.xlabel('Mass Position')
plt.ylabel('Relative Displacement')
plt.grid(True)
plt.xticks([0, 1, 2], ['Fixed', 'Mass 1', 'Mass 2'])

plt.tight_layout()
plt.show()


# %% ####################################################################################
# Time Domain Analysis

Arb, Brb = state_space_form(M, K, C)
print(f'Arb:\n{Arb}')
print(f'\nBrb:\n{Brb}')

# Force parameters
omega1 = 1.0
epsilon1 = 0.0
FA1 = 6.0
FC1 = 6.0

# Frequency range for FRF calculation
freq_range = np.linspace(0.1, 3.0, 1000)
H = calculate_frf(freq_range, M, K, C)

# Plot FRF magnitude
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.semilogy(freq_range, np.abs(H[:, 0, 0]), 'b-', linewidth=2)
plt.axvline(natural_frequencies_rad[0], color='r', linestyle='--', alpha=0.7, label=f'Mode 1: {natural_frequencies_hz[0]:.3f} Hz')
plt.axvline(natural_frequencies_rad[1], color='r', linestyle='--', alpha=0.7, label=f'Mode 2: {natural_frequencies_hz[1]:.3f} Hz')
plt.axvline(omega1, color='g', linestyle=':', alpha=0.7, label=f'Forcing freq: {omega1/(2*np.pi):.3f} Hz')
plt.title('FRF: Force at Mass 1 → Displacement of Mass 1')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.semilogy(freq_range, np.abs(H[:, 1, 0]), 'b-', linewidth=2)
plt.axvline(natural_frequencies_rad[0], color='r', linestyle='--', alpha=0.7, label=f'Mode 1: {natural_frequencies_hz[0]:.3f} Hz')
plt.axvline(natural_frequencies_rad[1], color='r', linestyle='--', alpha=0.7, label=f'Mode 2: {natural_frequencies_hz[1]:.3f} Hz')
plt.axvline(omega1, color='g', linestyle=':', alpha=0.7, label=f'Forcing freq: {omega1/(2*np.pi):.3f} Hz')
plt.title('FRF: Force at Mass 1 → Displacement of Mass 2')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.semilogy(freq_range, np.abs(H[:, 0, 1]), 'b-', linewidth=2)
plt.axvline(natural_frequencies_rad[0], color='r', linestyle='--', alpha=0.7, label=f'Mode 1: {natural_frequencies_hz[0]:.3f} Hz')
plt.axvline(natural_frequencies_rad[1], color='r', linestyle='--', alpha=0.7, label=f'Mode 2: {natural_frequencies_hz[1]:.3f} Hz')
plt.axvline(omega1, color='g', linestyle=':', alpha=0.7, label=f'Forcing freq: {omega1/(2*np.pi):.3f} Hz')
plt.title('FRF: Force at Mass 2 → Displacement of Mass 1')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.semilogy(freq_range, np.abs(H[:, 1, 1]), 'b-', linewidth=2)
plt.axvline(natural_frequencies_rad[0], color='r', linestyle='--', alpha=0.7, label=f'Mode 1: {natural_frequencies_hz[0]:.3f} Hz')
plt.axvline(natural_frequencies_rad[1], color='r', linestyle='--', alpha=0.7, label=f'Mode 2: {natural_frequencies_hz[1]:.3f} Hz')
plt.axvline(omega1, color='g', linestyle=':', alpha=0.7, label=f'Forcing freq: {omega1/(2*np.pi):.3f} Hz')
plt.title('FRF: Force at Mass 2 → Displacement of Mass 2')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Check if forcing frequency is close to natural frequencies
print('\n=== RESONANCE ANALYSIS ===')
for i, (nat_freq_rad, nat_freq_hz) in enumerate(zip(natural_frequencies_rad, natural_frequencies_hz)):
    ratio = omega1 / nat_freq_rad
    print(f'Forcing frequency / Natural frequency {i+1}: {ratio:.3f}')
    if 0.8 <= ratio <= 1.2:
        print(f'  ⚠️  WARNING: Forcing frequency is close to Mode {i+1}! Potential resonance.')
    elif ratio < 0.5:
        print(f'  ✓ Forcing frequency is well below Mode {i+1} (quasi-static regime)')
    elif ratio > 2.0:
        print(f'  ✓ Forcing frequency is well above Mode {i+1} (inertial regime)')
    else:
        print(f'  ⚠️  Forcing frequency is in the dynamic range of Mode {i+1}')

# Define the external force
F = lambda t: np.array([FA1*np.sin(omega1*t + epsilon1) + FC1, 
                        0.8*FA1*np.cos(omega1*t + epsilon1) + 0.8*FC1])

Equation = lambda t, x: Arb @ x + Brb @ F(t)

t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
x0 = np.array([0.0, 0.0, 0.0, 0.0])  # initial conditions

print('\nExternal force plot:')
plt.figure(figsize=(10, 4))
plt.plot(t_eval, F(t_eval)[0], label='Force at mass 1')
plt.plot(t_eval, F(t_eval)[1], label='Force at mass 2')
plt.title('Force')
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Solve the equation
solution = solve_ivp(Equation, t_span, x0, t_eval=t_eval)

# Extract positions
x1 = solution.y[0]
x2 = solution.y[1]

# Plot the result
plt.figure(figsize=(10, 4))
plt.plot(solution.t, x1, label='Mass 1 (x1)')
plt.plot(solution.t, x2, label='Mass 2 (x2)', linestyle='--')
plt.title('Dual Oscillator Response')
plt.xlabel('Time [s]')
plt.ylabel('Displacement [m]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% ####################################################################################
# Simulator and Sampler Functions

def simulator(FA1: float, FC1: float, omega1: float = 1.0, 
              k1: float = k1, k2: float = k2, k3: float = k3,
              t_span: Tuple[float, float] = (0, 500), n_points: int = 1000) -> Dict:
    """
    Run the dual oscillator simulation with given force parameters and spring constants
    
    Parameters:
    -----------
    FA1: float
        Amplitude for harmonic force
    FC1: float
        Constant force component (applied to both masses)
    omega1: float
        Frequency of harmonic force (default: 1.0)
    k1, k2, k3: float
        Spring constants
    t_span: tuple
        Time span for simulation (start, end)
    n_points: int
        Number of points to evaluate
        
    Returns:
    --------
    dict: Dictionary containing:
        - time series data (x1, x2)
        - statistical features (means, stds)
        - force parameters used
    """
    # Generate random phase shift
    epsilon1 = np.random.uniform(0, 2*np.pi)
    
    # System matrices with variable spring constants
    M_sim = np.array([[m1, 0], [0, m2]])
    K_sim = np.array([[k1+k2, -k2], [-k2, k2+k3]])
    C_sim = np.array([[c1+c2, -c2], [-c2, c2+c3]])
    
    Arb_sim, Brb_sim = state_space_form(M_sim, K_sim, C_sim)
    
    # Define force function
    F = lambda t: np.array([FA1*np.sin(omega1*t + epsilon1) + FC1, 
                            0.8*FA1*np.cos(omega1*t + epsilon1) + 0.8*FC1])
    
    Equation = lambda t, x: Arb_sim @ x + Brb_sim @ F(t)
    
    # Time points
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Solve
    solution = solve_ivp(Equation, t_span, x0, t_eval=t_eval)
    
    # Extract positions
    x1 = solution.y[0]  # position of mass 1
    x2 = solution.y[1]  # position of mass 2
    
    # Calculate statistical features (using steady-state region)
    steady_start = int(0.8 * len(x1))  # Use last 20% for steady-state
    features = {
        'x1_mean': np.mean(x1[steady_start:]),
        'x2_mean': np.mean(x2[steady_start:]),
        'x1_std': np.std(x1[steady_start:]),
        'x2_std': np.std(x2[steady_start:]),
        'params': {
            'FA1': FA1,
            'FC1': FC1,
            'omega1': omega1,
            'epsilon1': epsilon1,
            'k1': k1,
            'k2': k2,
            'k3': k3
        },
        'time_series': {
            't': solution.t,
            'x1': x1,
            'x2': x2
        }
    }
    
    return features


def dual_oscillator_sampler(param_ranges: Optional[Dict] = None) -> Callable:
    """
    Create a sampler function for dual oscillator parameters.
    
    Parameters:
    -----------
    param_ranges: dict, optional
        Dictionary with parameter names as keys and (min, max) tuples as values.
        Default ranges:
            - FA1: (0.5, 6.0)
            - FC1: (0.5, 6.0)
            - omega1: (0.1, 1.0)
    
    Returns:
    --------
    function: A sampler function that takes n_samples and returns a list of parameter dicts
    
    Example:
    --------
    >>> sampler = dual_oscillator_sampler()
    >>> samples = sampler(10)
    >>> result = simulator(**samples[0])
    """
    if param_ranges is None:
        param_ranges = {
            'FA1': (0.5, 6.0),
            'FC1': (0.5, 6.0),
            'omega1': (0.1, 1.0)
        }
    
    def f(n_samples: int, random_state: Optional[int] = None) -> list:
        """
        Generate random parameter sets for the simulation.
        
        Parameters:
        -----------
        n_samples: int
            Number of parameter sets to generate
        random_state: int, optional
            Random seed for reproducibility
        
        Returns:
        --------
        list: List of dictionaries, each containing parameter values
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        samples = []
        for _ in range(n_samples):
            params = {
                key: np.random.uniform(val[0], val[1]) 
                for key, val in param_ranges.items()
            }
            samples.append(params)
        
        return samples
    
    return f


# %% ####################################################################################
# Generate and analyze samples

# Create sampler
sampler = dual_oscillator_sampler()

# Generate sample parameters
n_samples = 200
param_sets = sampler(n_samples, random_state=42)

# Collect results
results = []
for params in param_sets:
    sim_result = simulator(**params)
    results.append(sim_result)

# Plot statistical features
plt.figure(figsize=(12, 6))
plt.rcParams.update({'font.size': 18})

plt.subplot(2, 1, 1)
x1_means = [r['x1_mean'] for r in results]
x2_means = [r['x2_mean'] for r in results]
plt.plot(range(n_samples), x1_means, 'o-', label='Mass 1 mean', markersize=3)
plt.plot(range(n_samples), x2_means, 'o-', label='Mass 2 mean', markersize=3)
plt.title('Mean Displacements Across Samples')
plt.legend()
plt.grid(True)
plt.xticks(range(0, n_samples, 20))
plt.xlabel('Sample Number')
plt.ylabel('Displacement [m]')

plt.subplot(2, 1, 2)
x1_stds = [r['x1_std'] for r in results]
x2_stds = [r['x2_std'] for r in results]
plt.plot(range(n_samples), x1_stds, 'o-', label='Mass 1 std', markersize=3)
plt.plot(range(n_samples), x2_stds, 'o-', label='Mass 2 std', markersize=3)
plt.title('Displacement Standard Deviations Across Samples')
plt.legend()
plt.grid(True)
plt.xticks(range(0, n_samples, 20))
plt.xlabel('Sample Number')
plt.ylabel('Displacement [m]')

plt.tight_layout()
plt.show()


# %% ####################################################################################
# Degradation Simulation

def simulate_degradation(k1_initial: float, k2_initial: float, k3_initial: float,
                         degradation_rate: float = 0.1) -> Tuple[float, float, float]:
    """
    Simulate spring degradation by reducing spring constants
    
    Parameters:
    -----------
    k1_initial, k2_initial, k3_initial: float
        Initial spring constants
    degradation_rate: float
        Rate of degradation (0-1)
        
    Returns:
    --------
    tuple: (k1_degraded, k2_degraded, k3_degraded)
    """
    k1_degraded = k1_initial * (1 - degradation_rate)
    k2_degraded = k2_initial * (1 - degradation_rate)
    k3_degraded = k3_initial * (1 - degradation_rate)
    return k1_degraded, k2_degraded, k3_degraded


# Generate new data with degraded springs
degradation_rate = 0.3  # 30% degradation
k1_degraded, k2_degraded, k3_degraded = simulate_degradation(k1, k2, k3, degradation_rate)

print(f'\n=== DEGRADATION ANALYSIS ===')
print(f'Degradation rate: {degradation_rate*100}%')
print(f'Original springs: k1={k1:.3f}, k2={k2:.3f}, k3={k3:.3f}')
print(f'Degraded springs: k1={k1_degraded:.3f}, k2={k2_degraded:.3f}, k3={k3_degraded:.3f}')

# Generate new samples with degraded springs
n_new_samples = 10
new_param_sets = sampler(n_new_samples, random_state=123)

# Collect results with degraded springs
new_results = []
for params in new_param_sets:
    sim_result = simulator(**params, k1=k1_degraded, k2=k2_degraded, k3=k3_degraded)
    new_results.append(sim_result)


# %% ####################################################################################
# Metrics and Analysis Functions

def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate and print evaluation metrics.
    
    Parameters:
    -----------
    y_true: array
        True values
    y_pred: array
        Predicted values
    y_std: array, optional
        Prediction standard deviations
        
    Returns:
    --------
    dict: Dictionary of metrics
    """
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


# Compare original vs degraded results
print('\n=== STATISTICAL COMPARISON ===')
original_x1_means = [r['x1_mean'] for r in results]
original_x2_means = [r['x2_mean'] for r in results]
original_x1_stds = [r['x1_std'] for r in results]
original_x2_stds = [r['x2_std'] for r in results]

new_x1_means = [r['x1_mean'] for r in new_results]
new_x2_means = [r['x2_mean'] for r in new_results]
new_x1_stds = [r['x1_std'] for r in new_results]
new_x2_stds = [r['x2_std'] for r in new_results]

print(f'Original Cases (n={len(original_x1_means)}):')
print(f'  Mass 1 - Mean: {np.mean(original_x1_means):.4f} ± {np.std(original_x1_means):.4f}, Avg Std: {np.mean(original_x1_stds):.4f}')
print(f'  Mass 2 - Mean: {np.mean(original_x2_means):.4f} ± {np.std(original_x2_means):.4f}, Avg Std: {np.mean(original_x2_stds):.4f}')

print(f'\nNew Cases (n={len(new_x1_means)}):')
print(f'  Mass 1 - Mean: {np.mean(new_x1_means):.4f} ± {np.std(new_x1_means):.4f}, Avg Std: {np.mean(new_x1_stds):.4f}')
print(f'  Mass 2 - Mean: {np.mean(new_x2_means):.4f} ± {np.std(new_x2_means):.4f}, Avg Std: {np.mean(new_x2_stds):.4f}')

print(f'\nDifferences (New - Original):')
print(f'  Mass 1 Mean shift: {np.mean(new_x1_means) - np.mean(original_x1_means):.4f}')
print(f'  Mass 2 Mean shift: {np.mean(new_x2_means) - np.mean(original_x2_means):.4f}')
print(f'  Mass 1 Std change: {np.mean(new_x1_stds) - np.mean(original_x1_stds):.4f}')
print(f'  Mass 2 Std change: {np.mean(new_x2_stds) - np.mean(original_x2_stds):.4f}')

# Visualize comparison
plt.figure(figsize=(15, 10))

# Plot 1: Comparison of mean PDFs
plt.subplot(2, 2, 1)
x_range_x1 = np.linspace(
    min(min(original_x1_means) - 3*max(original_x1_stds), 
        min(new_x1_means) - 3*max(new_x1_stds)),
    max(max(original_x1_means) + 3*max(original_x1_stds), 
        max(new_x1_means) + 3*max(new_x1_stds)),
    1000
)

mean_orig_x1_mean = np.mean(original_x1_means)
mean_orig_x1_std = np.mean(original_x1_stds)
mean_new_x1_mean = np.mean(new_x1_means)
mean_new_x1_std = np.mean(new_x1_stds)

plt.plot(x_range_x1, norm.pdf(x_range_x1, mean_orig_x1_mean, mean_orig_x1_std), 
         'b-', linewidth=3, label=f'Original (μ={mean_orig_x1_mean:.3f}, σ={mean_orig_x1_std:.3f})')
plt.plot(x_range_x1, norm.pdf(x_range_x1, mean_new_x1_mean, mean_new_x1_std), 
         'r-', linewidth=3, label=f'Degraded (μ={mean_new_x1_mean:.3f}, σ={mean_new_x1_std:.3f})')
plt.title('Mass 1 Displacement PDFs')
plt.xlabel('Displacement [m]')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Comparison for Mass 2
plt.subplot(2, 2, 2)
x_range_x2 = np.linspace(
    min(min(original_x2_means) - 3*max(original_x2_stds), 
        min(new_x2_means) - 3*max(new_x2_stds)),
    max(max(original_x2_means) + 3*max(original_x2_stds), 
        max(new_x2_stds) + 3*max(new_x2_stds)),
    1000
)

mean_orig_x2_mean = np.mean(original_x2_means)
mean_orig_x2_std = np.mean(original_x2_stds)
mean_new_x2_mean = np.mean(new_x2_means)
mean_new_x2_std = np.mean(new_x2_stds)

plt.plot(x_range_x2, norm.pdf(x_range_x2, mean_orig_x2_mean, mean_orig_x2_std), 
         'b-', linewidth=3, label=f'Original (μ={mean_orig_x2_mean:.3f}, σ={mean_orig_x2_std:.3f})')
plt.plot(x_range_x2, norm.pdf(x_range_x2, mean_new_x2_mean, mean_new_x2_std), 
         'r-', linewidth=3, label=f'Degraded (μ={mean_new_x2_mean:.3f}, σ={mean_new_x2_std:.3f})')
plt.title('Mass 2 Displacement PDFs')
plt.xlabel('Displacement [m]')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Histogram comparison for Mass 1
plt.subplot(2, 2, 3)
plt.hist(original_x1_means, bins=20, alpha=0.6, color='blue', label='Original x1_mean', density=True)
plt.hist(new_x1_means, bins=20, alpha=0.6, color='red', label='Degraded x1_mean', density=True)
plt.axvline(mean_orig_x1_mean, color='blue', linestyle='--', linewidth=2, label=f'Original μ={mean_orig_x1_mean:.3f}')
plt.axvline(mean_new_x1_mean, color='red', linestyle='--', linewidth=2, label=f'Degraded μ={mean_new_x1_mean:.3f}')
plt.title('Distribution of Mass 1 Mean Displacements')
plt.xlabel('Mean Displacement [m]')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Histogram comparison for Mass 2
plt.subplot(2, 2, 4)
plt.hist(original_x2_means, bins=20, alpha=0.6, color='blue', label='Original x2_mean', density=True)
plt.hist(new_x2_means, bins=20, alpha=0.6, color='red', label='Degraded x2_mean', density=True)
plt.axvline(mean_orig_x2_mean, color='blue', linestyle='--', linewidth=2, label=f'Original μ={mean_orig_x2_mean:.3f}')
plt.axvline(mean_new_x2_mean, color='red', linestyle='--', linewidth=2, label=f'Degraded μ={mean_new_x2_mean:.3f}')
plt.title('Distribution of Mass 2 Mean Displacements')
plt.xlabel('Mean Displacement [m]')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% ####################################################################################
