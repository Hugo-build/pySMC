################################################################################
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field 
from typing import Protocol, Dict, List, Any, Tuple, Optional, Union, Callable, Literal
from enum import Enum

import jax.numpy as jnp
import numpy as np
Array = np.ndarray

from .Aquiz import AcquizFunc, VarianceMin
from .Variables import VariableSet
from .Weighted import WeightStrategy, SizeNoveltyWeight
from .DataWash import train_test_split

# Optional imports for backends (will fail gracefully if not installed)
try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
################################################################################


"""
Universal Surrogate Modeling Framework for pySMC.

This module provides a flexible adapter framework for surrogate models with support
for adaptive sampling strategies (adaptive kriging, active learning, etc.).

Architecture:
    - BaseSurrogate: Core interface for all surrogates
    - AdaptiveSurrogate: Adds adaptive sampling capabilities
    - Model-specific adapters: Wrap concrete implementations (GP, NN, etc.)
    - Acquisition functions: Strategies for selecting next sample points
    - Data type utilities: Handle numpy/jax/torch conversions seamlessly

___________________________________________________________________________________________________________

ASCII overview (pipe emits surrogate "balls" into a pool):

    Raw X,y                 Thin data pipe                        Surrogates              Pool
  --------------     -----------------------------------     --------------------    -----------------
  |  Data gen  | ==> | SurrogatePipe (scale, validate, | ==>    o   o   o   o     => | SurrogatePool |
  --------------     | postprocess, type-interop)      |        ^   ^   ^   ^        -----------------
                     -----------------------------------       /   /   /   /
                                 ^                            /   /   /   /
                                 |                           /   /   /   /
                         model.predict(...)    <------------'   /   /   /
                                                               /   /   /
    Backends (examples):                                      /   /   /
       - sklearn.GPRegressor --------------------------------'   /   /
       - JAX GP (GPax.GaussianProcess) -------------------------'   /
       - Future NN/other models -----------------------------------'
___________________________________________________________________________________________________________

NOTE:
  - The pipe never trains; it only prepares inputs/outputs and wraps predict I/O.
  - Each model owns its own fit/predict; `make_predict_fn` bridges to adaptive hooks.
  - The pool stores/organizes multiple pipes and can implement selection/ensembles.


TODO:
- [ ] Add a way to save and load the surrogate pipe and pool.
- [ ] Add a general way to have a fit_fn
- [ ] Add a general way to have a predict_fn
"""




# =========================
# 0) Data Type Utilities
# =========================

def to_numpy(array: Union[np.ndarray, jnp.ndarray, Any]) -> np.ndarray:
    """
    Convert any array-like to numpy array.
    
    Handles:
    - numpy arrays (returns as-is)
    - JAX arrays (converts to numpy)
    - PyTorch tensors (converts to numpy)
    - Python lists (converts to numpy)
    
    Args:
        array: Input array in any supported format
    
    Returns:
        numpy.ndarray
    """
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, jnp.ndarray):
        return np.asarray(array)
    elif TORCH_AVAILABLE and torch is not None and isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    else:
        return np.asarray(array)


def to_jax(array: Union[np.ndarray, jnp.ndarray, Any]) -> jnp.ndarray:
    """
    Convert any array-like to JAX array.
    
    Args:
        array: Input array in any supported format
    
    Returns:
        jax.numpy.ndarray
    """
    if isinstance(array, jnp.ndarray):
        return array
    elif isinstance(array, np.ndarray):
        return jnp.asarray(array)
    elif TORCH_AVAILABLE and torch is not None and isinstance(array, torch.Tensor):
        return jnp.asarray(array.detach().cpu().numpy())
    else:
        return jnp.asarray(array)


def detect_array_type(array: Union[np.ndarray, jnp.ndarray, Any]) -> str:
    """
    Detect the type of array backend.
    
    Returns:
        'numpy', 'jax', 'torch', or 'unknown'
    """
    if isinstance(array, np.ndarray):
        return 'numpy'
    elif isinstance(array, jnp.ndarray):
        return 'jax'
    elif TORCH_AVAILABLE and torch is not None and isinstance(array, torch.Tensor):
        return 'torch'
    else:
        return 'unknown'


# =======================================================
# Adaptive Learning Utility Functions
# =======================================================

def calc_upd_weight(
    X_old: np.ndarray,
    y_old: np.ndarray,
    X_new: np.ndarray,
    y_new: np.ndarray,
    predict_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    strategy: Optional[WeightStrategy] = None,
    verbose: bool = False) -> float:
    """
    Calculate the adaptive weight for balancing old and new data.
    
    This function uses a WeightStrategy to determine how to balance old and new data.
    By default, uses SizeNoveltyWeight strategy (SS * NS^0.5).
    
    Args:
        X_old: Old/existing input data (N_old, D)
        y_old: Old/existing output data (N_old,)
        X_new: New input data (N_new, D)
        y_new: New output data (N_new,)
        predict_fn: Function that predicts (mean, std) for given X
        strategy: WeightStrategy instance (default: SizeNoveltyWeight)
        verbose: Whether to print debug information
    
    Returns:
        w_new_data: Weight for new data in range [0, 1]
    """
    # Use default strategy if none provided
    if strategy is None:
        strategy = SizeNoveltyWeight(novelty_power=0.5)
    
    # Calculate weight using the strategy
    weight = strategy.calculate(
        X_old=X_old,
        y_old=y_old,
        X_new=X_new,
        y_new=y_new,
        predict_fn=predict_fn
    )
    
    if verbose:
        strategy_name = strategy.__class__.__name__
        print(f"Weight Strategy: {strategy_name}")
        print(f"Calculated weight: {weight:.4f}")
        
        # If using SizeNoveltyWeight, show detailed breakdown
        if isinstance(strategy, SizeNoveltyWeight):
            N_old = X_old.shape[0]
            N_new = X_new.shape[0]
            SS = np.clip(N_new / N_old if N_old > 0 else 1.0, 0.0, 1.0)
            
            mean_pred, _ = predict_fn(X_new)
            mean_pred_np = to_numpy(mean_pred)
            ss_res = np.sum((y_new - mean_pred_np) ** 2)
            ss_tot = np.sum((y_new - np.mean(y_new)) ** 2)
            r2 = np.clip(1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 1.0, 0.0, 1.0)
            NS = 1.0 - r2
            
            print(f"  SS (Size Score): {SS:.4f}")
            print(f"  NS (Novelty Score - 1-R²): {NS:.4f}")
    
    return weight


def combine_weighted_data(
    X_old: np.ndarray,
    y_old: np.ndarray,
    X_new: np.ndarray,
    y_new: np.ndarray,
    weight: float,
    random_state: Optional[int] = None,
    verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine old and new data using weighted sampling.
    
    Samples from old and new data based on the provided weight:
    - proportion_new = weight (clipped to [0, 1])
    - proportion_old = 1 - proportion_new
    
    Args:
        X_old: Old/existing input data (N_old, D)
        y_old: Old/existing output data (N_old,)
        X_new: New input data (N_new, D)
        y_new: New output data (N_new,)
        weight: Weight for new data in range [0, 1]
        random_state: Random seed for sampling
        verbose: Whether to print debug information
    
    Returns:
        X_combined: Combined input data (N_combined, D)
        y_combined: Combined output data (N_combined,)
    """
    N_old = X_old.shape[0]
    N_new = X_new.shape[0]
    
    # Determine proportions for sampling
    proportion_new = np.clip(weight, 0.0, 1.0)
    proportion_old = 1.0 - proportion_new
    
    num_new_to_sample = int(proportion_new * N_new)
    num_old_to_sample = int(proportion_old * N_old)
    
    # Adjust sample counts if both are zero but data exists
    if num_new_to_sample == 0 and num_old_to_sample == 0:
        if N_new > 0 and N_old > 0:
            if weight >= 0.5:
                num_new_to_sample = 1
            else:
                num_old_to_sample = 1
        elif N_new > 0:
            num_new_to_sample = 1
        elif N_old > 0:
            num_old_to_sample = 1
    
    if verbose:
        print(f"Sampling {num_old_to_sample}/{N_old} old samples, {num_new_to_sample}/{N_new} new samples")
    
    # Sample from old and new data
    rng = np.random.default_rng(random_state)
    
    # Sample from new data
    if N_new > 0 and num_new_to_sample > 0:
        idx_new = rng.choice(N_new, size=num_new_to_sample, replace=False)
        X_new_sampled = X_new[idx_new]
        y_new_sampled = y_new[idx_new]
    else:
        X_new_sampled = np.empty((0, X_new.shape[1]))
        y_new_sampled = np.empty((0,))
    
    # Sample from old data
    if N_old > 0 and num_old_to_sample > 0:
        idx_old = rng.choice(N_old, size=num_old_to_sample, replace=False)
        X_old_sampled = X_old[idx_old]
        y_old_sampled = y_old[idx_old]
    else:
        X_old_sampled = np.empty((0, X_old.shape[1]))
        y_old_sampled = np.empty((0,))
    
    # Combine sampled data
    if X_new_sampled.shape[0] == 0 and X_old_sampled.shape[0] == 0:
        if verbose:
            print("Warning: Combined dataset is empty.")
        return np.empty((0, X_new.shape[1])), np.empty((0,))
    
    X_combined = np.vstack([X_old_sampled, X_new_sampled])
    y_combined = np.concatenate([y_old_sampled, y_new_sampled])
    
    return X_combined, y_combined





# =======================================================
# i.a) Scaler Protocol
# =======================================================

@dataclass 
class Scaler(Protocol):
    """
    Abstract base class for all scalers.
    """
    def fit(self, X:Array, y:Array) -> Scaler:
        """
        Fit the scaler to the data.
        """
        ...
    def transform(self, X:Array) -> Array:
        """
        Transform the data.
        """
        ...
    def inverse_transform(self, X:Array) -> Array:
        """
        Inverse transform the data.
        """
        ...

@dataclass
class StandardScaler:
    """
    Standard scaler.
    """
    mean_: Optional[Array] = None
    scale_: Optional[Array] = None
    eps_: float = 1e-6

    def fit(self, X:Array) -> StandardScaler:
       """
       Fit the standard scaler to the data.
       """
       if X.ndim == 1:
           X = X.reshape(-1, 1)
       self.mean_ = X.mean(axis=0)
       self.scale_ = X.std(axis=0) + self.eps_
       return self

    def transform(self, X:Array) -> Array:
       """
       Transform the data.
       """
       if X.ndim == 1:
           X = X.reshape(-1, 1)
       return (X - self.mean_) / self.scale_

    def inverse_transform(self, X:Array) -> Array:
       """
       Inverse transform the data.
       """
       if X.ndim == 1:
           X = X.reshape(-1, 1)
       return X * self.scale_ + self.mean_





# =======================================================
# i.b) Base Surrogate Protocol
# =======================================================


@dataclass
class SurrogatePipe:
    """
    Surrogate model pipeline.
    """
    model: object
    varSet: Optional[VariableSet] = None
    
    X: Optional[Array] = None
    y: Optional[Array] = None

    X_train: Optional[Array] = None
    y_train: Optional[Array] = None
    X_test: Optional[Array] = None
    y_test: Optional[Array] = None

    x_scaler: Optional[Scaler] = None
    y_scaler: Optional[Scaler] = None

    _scaled4X: bool = False
    _scaled4y: bool = False

    verbose: bool = False
    _fitted: bool = False


    def __post_init__(self):
        """
        Post-initialization setup.
        
        Note: Only fits scalers if they haven't been fitted yet. If you pass in
        already-fitted scalers, they will be used as-is.
        """
        # Check if x_scaler is already fitted (has mean_ attribute)
        if self.x_scaler is not None:
            if getattr(self.x_scaler, 'mean_', None) is not None:
                # Scaler already fitted, just mark as scaled
                self._scaled4X = True
            elif hasattr(self.model, 'X') and getattr(self.model, 'X') is not None:
                # Scaler not fitted, fit it on model's training data
                self.x_scaler.fit(to_numpy(self.model.X))
                self._scaled4X = True
        
        # Check if y_scaler is already fitted (has mean_ attribute)
        if self.y_scaler is not None:
            if getattr(self.y_scaler, 'mean_', None) is not None:
                # Scaler already fitted, just mark as scaled
                self._scaled4y = True
            elif hasattr(self.model, 'y') and getattr(self.model, 'y') is not None:
                # Scaler not fitted, fit it on model's training data
                self.y_scaler.fit(to_numpy(self.model.y))
                self._scaled4y = True

    # ------------------------------------------------------------
    # Configuration helpers for non initialized surrogate pipe
    # ------------------------------------------------------------
    def attach_model(self, model: object) -> None:
        self.model = model

    def set_scalers(self, x_scaler: Optional[Scaler] = None, y_scaler: Optional[Scaler] = None) -> None:
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self._scaled4X = x_scaler is not None and getattr(x_scaler, 'mean_', None) is not None
        self._scaled4y = y_scaler is not None and getattr(y_scaler, 'mean_', None) is not None

    def train_test_split(self, X: Array, y: Array, test_size: float = 0.2, random_state: Optional[int] = None) -> Tuple[Array, Array, Array, Array]:
        """
        Split the data into training and testing sets.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    
    
    # -----------------------
    # Validation utilities
    # -----------------------
    def validate_X(self, X: Array) -> None:
        X_np = to_numpy(X)
        if X_np.ndim != 2:
            raise ValueError("X must be 2D (N, D)")
        if not np.isfinite(X_np).all():
            raise ValueError("X contains NaN or Inf")

    def validate_y(self, y: Array) -> None:
        y_np = to_numpy(y)
        if y_np.ndim not in (1, 2):
            raise ValueError("y must be 1D or 2D")
        if not np.isfinite(y_np).all():
            raise ValueError("y contains NaN or Inf")

    # -----------------------
    # Transform utilities
    # -----------------------
    def transform_X(self, X: Array) -> Array:
        """
        Apply X scaling if scaler is available.
        """
        if self.x_scaler is not None and self._scaled4X:
            return self.x_scaler.transform(to_numpy(X))
        return X
    
    def postprocess_prediction(self, mean: Array, std: Optional[Array] = None) -> Tuple[Array, Optional[Array]]:
        """
        Apply inverse y scaling if scaler is available.
        """
        mean_out = mean
        std_out = std
        
        if self.y_scaler is not None and self._scaled4y:
            mean_np = to_numpy(mean)
            if mean_np.ndim == 1:
                mean_np = mean_np.reshape(-1, 1)
            mean_out = self.y_scaler.inverse_transform(mean_np).flatten()
            
            # Scale uncertainty by the y_scaler's scale
            if std is not None:
                std_np = to_numpy(std)
                std_out = std_np * self.y_scaler.scale_[0]
        
        return mean_out, std_out

    def predict(self, X:Array, return_std:bool = False) -> Union[Array, Tuple[Array, Array]]:
        """
        Convenience wrapper that applies X-scaling before calling the model and
        y inverse-scaling after. Does not own training; relies on model's predict.
        
        Note: This method tries to handle different model backends:
        - GPax models: Always return (mean, std) tuple
        - sklearn models: Use return_std parameter
        """
        X_in = self.transform_X(X)
        
        # Try to detect the model type and call predict appropriately
        try:
            # Try sklearn-style predict with return_std parameter
            if return_std:
                out = self.model.predict(X_in, return_std=True)
                mean, std = out
            else:
                mean = self.model.predict(X_in, return_std=False)
                std = None
        except TypeError:
            # Fall back to GPax-style (always returns tuple)
            out = self.model.predict(X_in)
            if isinstance(out, tuple) and len(out) == 2:
                mean, std = out
            else:
                mean = out
                std = None
        
        mean_pp, std_pp = self.postprocess_prediction(mean, std)
        return (mean_pp, std_pp) if return_std else mean_pp



    # -----------------------
    # Serialization utilities
    # -----------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the surrogate pipeline to a dictionary.
        """
        return {
            'model': self.model,
            'x_scaler': self.x_scaler,
            'y_scaler': self.y_scaler,
            'scaledX': self._scaled4X,
            'scaledY': self._scaled4y,
            'verbose': self.verbose
        }
        
    def from_dict(self, data: Dict[str, Any]) -> SurrogatePipe:
        """
        Convert a dictionary to a surrogate pipeline.
        """
        return SurrogatePipe(
            model=data['model'],
            x_scaler=data['x_scaler'],
            y_scaler=data['y_scaler'],
            _scaled4X=data.get('scaledX', False),
            _scaled4y=data.get('scaledY', False),
            verbose=data.get('verbose', False)
        )

    # -----------------------
    # Prediction function maker (for adaptive hooks)
    # -----------------------
    def make_predict_fn(self, model: Optional[object] = None) -> Callable[[Array], Tuple[Array, Array]]:
        """
        Create a callable that maps X -> (mean, std) using this pipe's
        transform/postprocess with the given model (default: self.model).
        
        Returns a function that always returns (mean, std) tuple for consistency
        with adaptive learning utilities.
        """
        mdl = model if model is not None else self.model
        def _predict_fn(X: Array) -> Tuple[Array, Array]:
            X_in = self.transform_X(X)
            
            # Try to detect the model type and call predict appropriately
            try:
                # Try sklearn-style predict with return_std parameter
                out = mdl.predict(X_in, return_std=True)
                mean, std = out
            except TypeError:
                # Fall back to GPax-style (always returns tuple)
                out = mdl.predict(X_in)
                if isinstance(out, tuple) and len(out) == 2:
                    mean, std = out
                else:
                    # Model returns only mean, create dummy std
                    mean = out
                    std = jnp.zeros_like(mean)
            
            return self.postprocess_prediction(mean, std)
        return _predict_fn
   


class SurrogatePool:
    """
    Surrogate pool.
    """
    surrogates: List[SurrogatePipe]
    def __init__(self, surrogates: List[SurrogatePipe]):
        self.surrogates = surrogates
    def add(self, surrogate: SurrogatePipe) -> None:
        self.surrogates.append(surrogate)
    def remove(self, surrogate: SurrogatePipe) -> None:
        self.surrogates.remove(surrogate)
    def get(self, index: int) -> SurrogatePipe:
        return self.surrogates[index]



if __name__ == "__main__":
    """
    Run tests for the surrogate framework.
    
    For comprehensive tests, see: examples/test_surrogate_framework.py
    """
    print("Surrogate module loaded successfully.")
    print("For comprehensive tests, run: python examples/test_surrogate_framework.py")

