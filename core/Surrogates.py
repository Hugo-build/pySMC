################################################################################
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field 
from typing import Protocol, Dict, List, Any, Tuple, Optional, Union, Callable, Literal
from enum import Enum
import json
import sys
from pathlib import Path
import jax.numpy as jnp
import numpy as np
Array = np.ndarray

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
- [x] Add a way to save and load the surrogate pipe and pool.
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
    Standard scaler for normalizing data to zero mean and unit variance.
    
    Supports JSON serialization for model persistence.
    
    Usage:
        >>> scaler = StandardScaler()
        >>> scaler.fit(X_train)
        >>> X_scaled = scaler.transform(X_train)
        >>> scaler.save("scaler.json")
        >>> loaded_scaler = StandardScaler.load("scaler.json")
    """
    mean_: Optional[Array] = None
    scale_: Optional[Array] = None
    eps_: float = 1e-6

    def fit(self, X: Array) -> StandardScaler:
        """
        Fit the standard scaler to the data.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) + self.eps_
        return self

    def transform(self, X: Array) -> Array:
        """
        Transform the data.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: Array) -> Array:
        """
        Inverse transform the data.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X * self.scale_ + self.mean_

    # ===== Serialization Methods =====
    def to_dict(self) -> Dict[str, Any]:
        """Convert the scaler to a serializable dictionary."""
        return {
            "type": "StandardScaler",
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "scale": self.scale_.tolist() if self.scale_ is not None else None,
            "eps": self.eps_
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert scaler to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save scaler to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StandardScaler:
        """Create a StandardScaler from a dictionary."""
        return cls(
            mean_=np.array(data['mean']) if data.get('mean') is not None else None,
            scale_=np.array(data['scale']) if data.get('scale') is not None else None,
            eps_=data.get('eps', 1e-6)
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> StandardScaler:
        """Create a StandardScaler from a JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> StandardScaler:
        """Load a StandardScaler from a JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())





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

    # =========================================
    # Serialization utilities
    # =========================================
    
    def _detect_model_type(self) -> str:
        """Detect the model backend type for serialization."""
        model_class = self.model.__class__.__name__
        model_module = self.model.__class__.__module__
        
        # Check for GPax models
        if 'GPax' in model_module or model_class == 'GaussianProcess':
            return 'gpax'
        # Check for sklearn models
        elif 'sklearn' in model_module:
            return 'sklearn'
        elif 'torch' in model_module:
            return 'torch'
        else:
            return 'unknown'
    
    def _serialize_scaler(self, scaler: Optional[Scaler]) -> Optional[Dict[str, Any]]:
        """Serialize a scaler to dict if it has to_dict method."""
        if scaler is None:
            return None
        if hasattr(scaler, 'to_dict'):
            return scaler.to_dict()
        return None
    
    def to_dict(self, include_data: bool = True) -> Dict[str, Any]:
        """
        Convert the surrogate pipeline to a serializable dictionary.
        
        Args:
            include_data: Whether to include X, y, X_train, etc. (default True)
        
        Returns:
            Dictionary with all serializable pipeline components
        
        Note: The model must have a to_dict() method for JSON serialization.
              For GPax models, this is built-in. For sklearn models, 
              use the folder-based save() method instead.
        """
        result = {
            'version': '1.0',
            'model_type': self._detect_model_type(),
            'x_scaler': self._serialize_scaler(self.x_scaler),
            'y_scaler': self._serialize_scaler(self.y_scaler),
            'scaled4X': self._scaled4X,
            'scaled4y': self._scaled4y,
            'verbose': self.verbose,
            'fitted': self._fitted,
        }
        
        # Include model if it has to_dict method
        if hasattr(self.model, 'to_dict'):
            result['model'] = self.model.to_dict()
        else:
            result['model'] = None
            result['_model_warning'] = f"Model type '{type(self.model).__name__}' does not support JSON serialization"
        
        # Include data arrays
        if include_data:
            result['X'] = self.X.tolist() if self.X is not None else None
            result['y'] = self.y.tolist() if self.y is not None else None
            result['X_train'] = self.X_train.tolist() if self.X_train is not None else None
            result['y_train'] = self.y_train.tolist() if self.y_train is not None else None
            result['X_test'] = self.X_test.tolist() if self.X_test is not None else None
            result['y_test'] = self.y_test.tolist() if self.y_test is not None else None
        
        return result
    
    def to_json(self, include_data: bool = True, indent: int = 2) -> str:
        """Convert pipeline to JSON string."""
        return json.dumps(self.to_dict(include_data=include_data), indent=indent)
    
    def save(self, 
             filepath: Union[str, Path],
             as_folder: bool = False,
             include_data: bool = True) -> None:
        """
        Save the surrogate pipeline to disk.
        
        Args:
            filepath: Path to save location
                - If as_folder=False: path to .json file
                - If as_folder=True: path to folder
            as_folder: If True, saves as folder with separate files for model, scalers, data
            include_data: Whether to include training/test data
        
        Example:
            # Save as single JSON file
            >>> pipe.save("my_pipe.json")
            
            # Save as folder (better for large data)
            >>> pipe.save("my_pipe_folder", as_folder=True)
        """
        filepath = Path(filepath)
        
        if as_folder:
            self._save_as_folder(filepath, include_data)
        else:
            self._save_as_json(filepath, include_data)
    
    def _save_as_json(self, filepath: Path, include_data: bool) -> None:
        """Save pipeline as a single JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(self.to_json(include_data=include_data))
        print(f"✓ SurrogatePipe saved to '{filepath}'")
    
    def _save_as_folder(self, dirpath: Path, include_data: bool) -> None:
        """Save pipeline as a folder with separate files."""
        dirpath.mkdir(parents=True, exist_ok=True)
        
        # Save manifest
        manifest = {
            'version': '1.0',
            'model_type': self._detect_model_type(),
            'has_model': hasattr(self.model, 'to_dict'),
            'has_x_scaler': self.x_scaler is not None,
            'has_y_scaler': self.y_scaler is not None,
            'has_data': include_data,
            'scaled4X': self._scaled4X,
            'scaled4y': self._scaled4y,
            'verbose': self.verbose,
            'fitted': self._fitted,
        }
        with open(dirpath / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Save model if it supports serialization
        if hasattr(self.model, 'save'):
            self.model.save(dirpath / 'model.json')
        elif hasattr(self.model, 'to_dict'):
            with open(dirpath / 'model.json', 'w') as f:
                json.dump(self.model.to_dict(), f, indent=2)
        
        # Save scalers
        if self.x_scaler is not None and hasattr(self.x_scaler, 'save'):
            self.x_scaler.save(dirpath / 'x_scaler.json')
        if self.y_scaler is not None and hasattr(self.y_scaler, 'save'):
            self.y_scaler.save(dirpath / 'y_scaler.json')
        
        # Save data arrays
        if include_data:
            data = {}
            if self.X is not None:
                data['X'] = self.X.tolist()
            if self.y is not None:
                data['y'] = self.y.tolist()
            if self.X_train is not None:
                data['X_train'] = self.X_train.tolist()
            if self.y_train is not None:
                data['y_train'] = self.y_train.tolist()
            if self.X_test is not None:
                data['X_test'] = self.X_test.tolist()
            if self.y_test is not None:
                data['y_test'] = self.y_test.tolist()
            
            if data:
                with open(dirpath / 'data.json', 'w') as f:
                    json.dump(data, f, indent=2)
        
        print(f"✓ SurrogatePipe saved to folder '{dirpath}/'")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], model: Optional[object] = None) -> SurrogatePipe:
        """
        Create a SurrogatePipe from a dictionary.
        
        Args:
            data: Dictionary from to_dict()
            model: Pre-loaded model object. Required if model wasn't serialized.
        
        Returns:
            SurrogatePipe instance
        """
        # Reconstruct scalers
        x_scaler = None
        y_scaler = None
        
        if data.get('x_scaler') is not None:
            scaler_data = data['x_scaler']
            if scaler_data.get('type') == 'StandardScaler':
                x_scaler = StandardScaler.from_dict(scaler_data)
        
        if data.get('y_scaler') is not None:
            scaler_data = data['y_scaler']
            if scaler_data.get('type') == 'StandardScaler':
                y_scaler = StandardScaler.from_dict(scaler_data)
        
        # Reconstruct model if provided in data and model_type is 'gpax'
        loaded_model = model
        if loaded_model is None and data.get('model') is not None:
            model_type = data.get('model_type', 'unknown')
            if model_type == 'gpax':
                # Import GPax and reconstruct
                try:
                    from .GPax import GaussianProcess
                    loaded_model, _ = GaussianProcess.from_dict(data['model'])
                except ImportError:
                    raise ValueError("Cannot load GPax model - GPax module not available")
        
        if loaded_model is None:
            raise ValueError(
                "Model could not be loaded from data. "
                "Please provide a pre-loaded model via the 'model' parameter."
            )
        
        # Reconstruct data arrays
        X = np.array(data['X']) if data.get('X') is not None else None
        y = np.array(data['y']) if data.get('y') is not None else None
        X_train = np.array(data['X_train']) if data.get('X_train') is not None else None
        y_train = np.array(data['y_train']) if data.get('y_train') is not None else None
        X_test = np.array(data['X_test']) if data.get('X_test') is not None else None
        y_test = np.array(data['y_test']) if data.get('y_test') is not None else None
        
        # Create pipe without triggering __post_init__ scaler fitting
        pipe = cls.__new__(cls)
        pipe.model = loaded_model
        pipe.varSet = None
        pipe.X = X
        pipe.y = y
        pipe.X_train = X_train
        pipe.y_train = y_train
        pipe.X_test = X_test
        pipe.y_test = y_test
        pipe.x_scaler = x_scaler
        pipe.y_scaler = y_scaler
        pipe._scaled4X = data.get('scaled4X', False)
        pipe._scaled4y = data.get('scaled4y', False)
        pipe.verbose = data.get('verbose', False)
        pipe._fitted = data.get('fitted', False)
        
        return pipe
    
    @classmethod
    def from_json(cls, json_str: str, model: Optional[object] = None) -> SurrogatePipe:
        """Create a SurrogatePipe from a JSON string."""
        return cls.from_dict(json.loads(json_str), model=model)
    
    @classmethod
    def load(cls, filepath: Union[str, Path], model: Optional[object] = None) -> SurrogatePipe:
        """
        Load a SurrogatePipe from disk.
        
        Args:
            filepath: Path to saved pipeline (.json file or folder)
            model: Pre-loaded model object (required for sklearn models)
        
        Returns:
            SurrogatePipe instance
        
        Example:
            # Load from JSON file (GPax models auto-loaded)
            >>> pipe = SurrogatePipe.load("my_pipe.json")
            
            # Load from folder
            >>> pipe = SurrogatePipe.load("my_pipe_folder")
            
            # Load with external model (for sklearn)
            >>> from sklearn.gaussian_process import GaussianProcessRegressor
            >>> my_model = train_sklearn_model(X, y)
            >>> pipe = SurrogatePipe.load("my_pipe.json", model=my_model)
        """
        filepath = Path(filepath)
        
        if filepath.is_dir():
            return cls._load_from_folder(filepath, model)
        else:
            return cls._load_from_json(filepath, model)
    
    @classmethod
    def _load_from_json(cls, filepath: Path, model: Optional[object] = None) -> SurrogatePipe:
        """Load pipeline from a single JSON file."""
        with open(filepath, 'r') as f:
            pipe = cls.from_json(f.read(), model=model)
        print(f"✓ SurrogatePipe loaded from '{filepath}'")
        return pipe
    
    @classmethod
    def _load_from_folder(cls, dirpath: Path, model: Optional[object] = None) -> SurrogatePipe:
        """Load pipeline from a folder."""
        # Load manifest
        with open(dirpath / 'manifest.json', 'r') as f:
            manifest = json.load(f)
        
        # Load model
        loaded_model = model
        if loaded_model is None and manifest.get('has_model'):
            model_path = dirpath / 'model.json'
            if model_path.exists():
                model_type = manifest.get('model_type', 'unknown')
                if model_type == 'gpax':
                    from .GPax import GaussianProcess
                    loaded_model, _ = GaussianProcess.load(model_path)
                else:
                    # Try to load as generic dict
                    with open(model_path, 'r') as f:
                        model_data = json.load(f)
                    # User must provide model
                    if model is None:
                        raise ValueError(
                            f"Model type '{model_type}' requires you to provide a pre-loaded model."
                        )
        
        if loaded_model is None:
            raise ValueError("Could not load model. Please provide via 'model' parameter.")
        
        # Load scalers
        x_scaler = None
        y_scaler = None
        
        x_scaler_path = dirpath / 'x_scaler.json'
        if x_scaler_path.exists():
            x_scaler = StandardScaler.load(x_scaler_path)
        
        y_scaler_path = dirpath / 'y_scaler.json'
        if y_scaler_path.exists():
            y_scaler = StandardScaler.load(y_scaler_path)
        
        # Load data
        X = y = X_train = y_train = X_test = y_test = None
        data_path = dirpath / 'data.json'
        if data_path.exists():
            with open(data_path, 'r') as f:
                data = json.load(f)
            X = np.array(data['X']) if data.get('X') else None
            y = np.array(data['y']) if data.get('y') else None
            X_train = np.array(data['X_train']) if data.get('X_train') else None
            y_train = np.array(data['y_train']) if data.get('y_train') else None
            X_test = np.array(data['X_test']) if data.get('X_test') else None
            y_test = np.array(data['y_test']) if data.get('y_test') else None
        
        # Create pipe
        pipe = cls.__new__(cls)
        pipe.model = loaded_model
        pipe.varSet = None
        pipe.X = X
        pipe.y = y
        pipe.X_train = X_train
        pipe.y_train = y_train
        pipe.X_test = X_test
        pipe.y_test = y_test
        pipe.x_scaler = x_scaler
        pipe.y_scaler = y_scaler
        pipe._scaled4X = manifest.get('scaled4X', False)
        pipe._scaled4y = manifest.get('scaled4y', False)
        pipe.verbose = manifest.get('verbose', False)
        pipe._fitted = manifest.get('fitted', False)
        
        print(f"✓ SurrogatePipe loaded from folder '{dirpath}/'")
        return pipe

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
    Pool of surrogate model pipelines.
    
    Manages multiple SurrogatePipe instances and supports saving/loading
    the entire pool as a folder structure.
    
    Usage:
        >>> pool = SurrogatePool([pipe1, pipe2, pipe3])
        >>> pool.save("my_pool")  # Saves as folder
        >>> loaded_pool = SurrogatePool.load("my_pool")
    """
    surrogates: List[SurrogatePipe]
    
    def __init__(self, surrogates: Optional[List[SurrogatePipe]] = None):
        self.surrogates = surrogates if surrogates is not None else []
    
    def add(self, surrogate: SurrogatePipe) -> None:
        """Add a surrogate pipe to the pool."""
        self.surrogates.append(surrogate)
    
    def remove(self, surrogate: SurrogatePipe) -> None:
        """Remove a surrogate pipe from the pool."""
        self.surrogates.remove(surrogate)
    
    def get(self, index: int) -> SurrogatePipe:
        """Get a surrogate pipe by index."""
        return self.surrogates[index]
    
    def __len__(self) -> int:
        """Return the number of surrogates in the pool."""
        return len(self.surrogates)
    
    def __iter__(self):
        """Iterate over surrogates."""
        return iter(self.surrogates)
    
    def __getitem__(self, index: int) -> SurrogatePipe:
        """Get surrogate by index."""
        return self.surrogates[index]
    
    # ===== Serialization Methods =====
    def save(self, 
             dirpath: Union[str, Path],
             include_data: bool = True) -> None:
        """
        Save the entire pool as a folder structure.
        
        Structure:
            dirpath/
            ├── manifest.json      # Pool metadata
            ├── pipe_0/            # First pipe (folder)
            ├── pipe_1/            # Second pipe (folder)
            └── ...
        
        Args:
            dirpath: Path to the pool folder
            include_data: Whether to include training/test data for each pipe
        
        Example:
            >>> pool.save("my_surrogate_pool")
        """
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        
        # Save manifest
        manifest = {
            'version': '1.0',
            'num_surrogates': len(self.surrogates),
            'pipe_names': [f'pipe_{i}' for i in range(len(self.surrogates))],
        }
        with open(dirpath / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Save each pipe as a folder
        for i, pipe in enumerate(self.surrogates):
            pipe_dir = dirpath / f'pipe_{i}'
            pipe.save(pipe_dir, as_folder=True, include_data=include_data)
        
        print(f"✓ SurrogatePool saved to '{dirpath}/' ({len(self.surrogates)} pipes)")
    
    @classmethod
    def load(cls, 
             dirpath: Union[str, Path],
             models: Optional[List[object]] = None) -> SurrogatePool:
        """
        Load a SurrogatePool from a folder.
        
        Args:
            dirpath: Path to the pool folder
            models: Optional list of pre-loaded models (for sklearn, etc.)
                   Must match the number of pipes if provided.
        
        Returns:
            SurrogatePool instance with all pipes loaded
        
        Example:
            >>> pool = SurrogatePool.load("my_surrogate_pool")
            >>> print(f"Loaded {len(pool)} pipes")
        """
        dirpath = Path(dirpath)
        
        # Load manifest
        with open(dirpath / 'manifest.json', 'r') as f:
            manifest = json.load(f)
        
        num_surrogates = manifest['num_surrogates']
        pipe_names = manifest.get('pipe_names', [f'pipe_{i}' for i in range(num_surrogates)])
        
        # Validate models list if provided
        if models is not None and len(models) != num_surrogates:
            raise ValueError(
                f"Number of models ({len(models)}) doesn't match "
                f"number of pipes ({num_surrogates})"
            )
        
        # Load each pipe
        surrogates = []
        for i, pipe_name in enumerate(pipe_names):
            pipe_dir = dirpath / pipe_name
            model = models[i] if models is not None else None
            pipe = SurrogatePipe.load(pipe_dir, model=model)
            surrogates.append(pipe)
        
        pool = cls(surrogates)
        print(f"✓ SurrogatePool loaded from '{dirpath}/' ({len(surrogates)} pipes)")
        return pool
    
    def to_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the pool (without full serialization).
        
        Returns:
            Dictionary with pool summary info
        """
        summaries = []
        for i, pipe in enumerate(self.surrogates):
            summary = {
                'index': i,
                'model_type': pipe._detect_model_type(),
                'has_x_scaler': pipe.x_scaler is not None,
                'has_y_scaler': pipe.y_scaler is not None,
                'n_samples': pipe.X.shape[0] if pipe.X is not None else 0,
                'n_features': pipe.X.shape[1] if pipe.X is not None else 0,
            }
            summaries.append(summary)
        
        return {
            'num_surrogates': len(self.surrogates),
            'pipes': summaries
        }


