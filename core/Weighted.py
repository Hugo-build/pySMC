"""
Adaptive Weight Strategies for Surrogate Model Updates.

This module provides a template-based framework for calculating adaptive weights
when updating surrogate models with new data. Different strategies can balance
old and new data based on various criteria (novelty, size, uncertainty, etc.).

Design Pattern:
    - WeightStrategy: Abstract base class defining the interface
    - Concrete implementations: Various weight calculation strategies
    - Easy to extend: Users can create custom strategies by subclassing
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional
import numpy as np
import jax.numpy as jnp


# =======================================================
# Base Weight Strategy Interface
# =======================================================

class WeightStrategy(ABC):
    """
    Base class for adaptive weight calculation strategies.
    
    Weight strategies determine how to balance old (existing) and new data
    when updating a surrogate model. The weight represents the importance
    of new data relative to old data.
    
    Returns:
        weight: Float in range [0, 1]
            - 0.0 = use only old data
            - 1.0 = use only new data
            - 0.5 = equal balance
    """
    
    @abstractmethod
    def calculate(self,
                  X_old: np.ndarray,
                  y_old: np.ndarray,
                  X_new: np.ndarray,
                  y_new: np.ndarray,
                  predict_fn: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
                  **kwargs) -> float:
        """
        Calculate the adaptive weight for new data.
        
        Args:
            X_old: Old/existing input data (N_old, D)
            y_old: Old/existing output data (N_old,)
            X_new: New input data (N_new, D)
            y_new: New output data (N_new,)
            predict_fn: Function that predicts (mean, std) for given X
            **kwargs: Strategy-specific parameters
        
        Returns:
            weight: Importance weight for new data in [0, 1]
        """
        pass


# =======================================================
# Concrete Weight Strategy Implementations
# =======================================================

class UniformWeight(WeightStrategy):
    """
    Uniform weight strategy: Always use a fixed weight.
    
    Useful for:
    - Baseline comparisons
    - Simple fixed-ratio updates
    - Testing purposes
    """
    
    def __init__(self, weight: float = 0.5):
        """
        Args:
            weight: Fixed weight for new data (default: 0.5 = equal balance)
        """
        self.weight = np.clip(weight, 0.0, 1.0)
    
    def calculate(self,
                  X_old: np.ndarray,
                  y_old: np.ndarray,
                  X_new: np.ndarray,
                  y_new: np.ndarray,
                  predict_fn: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
                  **kwargs) -> float:
        """Return the fixed weight."""
        return self.weight


class SizeWeight(WeightStrategy):
    """
    Size-based weight: Weight proportional to relative dataset size.
    
    Weight = N_new / N_old (clipped to [0, 1])
    
    Useful for:
    - Simple adaptive updates
    - When data quantity matters more than quality
    """
    
    def calculate(self,
                  X_old: np.ndarray,
                  y_old: np.ndarray,
                  X_new: np.ndarray,
                  y_new: np.ndarray,
                  predict_fn: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
                  **kwargs) -> float:
        """Calculate weight based on dataset sizes."""
        N_old = X_old.shape[0]
        N_new = X_new.shape[0]
        
        if N_old == 0:
            return 1.0  # Only new data available
        
        size_ratio = N_new / N_old
        return np.clip(size_ratio, 0.0, 1.0)


class NoveltyWeight(WeightStrategy):
    """
    Novelty-based weight: Weight based on how novel new data is.
    
    Novelty Score (NS) = 1 - R²
    where R² measures how well current model predicts new data.
    
    Useful for:
    - Emphasizing new information
    - Adapting to distribution shifts
    - Active learning scenarios
    """
    
    def __init__(self, power: float = 1.0):
        """
        Args:
            power: Exponent for novelty score (default: 1.0)
                   - power < 1: Less sensitive to novelty
                   - power > 1: More sensitive to novelty
        """
        self.power = power
    
    def calculate(self,
                  X_old: np.ndarray,
                  y_old: np.ndarray,
                  X_new: np.ndarray,
                  y_new: np.ndarray,
                  predict_fn: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
                  **kwargs) -> float:
        """Calculate weight based on novelty (prediction error)."""
        if predict_fn is None:
            raise ValueError("NoveltyBasedWeight requires predict_fn")
        
        # Import to_numpy from Surrogates if available, otherwise define locally
        try:
            from Surrogates import to_numpy
        except ImportError:
            def to_numpy(arr):
                if isinstance(arr, np.ndarray):
                    return arr
                elif isinstance(arr, jnp.ndarray):
                    return np.asarray(arr)
                else:
                    return np.asarray(arr)
        
        # Predict on new data
        mean_pred, _ = predict_fn(X_new)
        mean_pred_np = to_numpy(mean_pred)
        
        # Calculate R²
        ss_res = np.sum((y_new - mean_pred_np) ** 2)
        ss_tot = np.sum((y_new - np.mean(y_new)) ** 2)
        
        if ss_tot < 1e-10:
            r2 = 1.0  # Perfect fit or no variance
        else:
            r2 = 1.0 - ss_res / ss_tot
        
        r2 = np.clip(r2, 0.0, 1.0)
        novelty = 1.0 - r2
        
        return np.clip(novelty ** self.power, 0.0, 1.0)


class SizeNoveltyWeight(WeightStrategy):
    """
    Combined Size-Novelty weight strategy (default in pySMC).
    
    This implements the adaptive learning strategy from test_dualOscillator.py:
    - Size Score (SS) = N_new / N_old (clipped to [0, 1])
    - Novelty Score (NS) = 1 - R²
    - Weight = SS * NS^power
    
    Useful for:
    - Balanced adaptive learning
    - Accounting for both quantity and quality of new data
    - Most general-purpose scenarios
    """
    
    def __init__(self, novelty_power: float = 0.5):
        """
        Args:
            novelty_power: Exponent for novelty score (default: 0.5)
                          - Controls sensitivity to novelty
                          - 0.5 = square root (moderate sensitivity)
                          - 1.0 = linear (high sensitivity)
                          - < 0.5 = sublinear (low sensitivity)
        """
        self.novelty_power = novelty_power
    
    def calculate(self,
                  X_old: np.ndarray,
                  y_old: np.ndarray,
                  X_new: np.ndarray,
                  y_new: np.ndarray,
                  predict_fn: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
                  **kwargs) -> float:
        """Calculate weight based on both size and novelty."""
        if predict_fn is None:
            raise ValueError("SizeNoveltyWeight requires predict_fn")
        
        # Import to_numpy from Surrogates if available, otherwise define locally
        try:
            from Surrogates import to_numpy
        except ImportError:
            def to_numpy(arr):
                if isinstance(arr, np.ndarray):
                    return arr
                elif isinstance(arr, jnp.ndarray):
                    return np.asarray(arr)
                else:
                    return np.asarray(arr)
        
        N_old = X_old.shape[0]
        N_new = X_new.shape[0]
        
        if N_old == 0:
            return 1.0  # Only new data available
        
        # Calculate Size Score (SS)
        SS = N_new / N_old
        SS = np.clip(SS, 0.0, 1.0)
        
        # Calculate Novelty Score (NS) using R²
        mean_pred, _ = predict_fn(X_new)
        mean_pred_np = to_numpy(mean_pred)
        
        # Calculate R²
        ss_res = np.sum((y_new - mean_pred_np) ** 2)
        ss_tot = np.sum((y_new - np.mean(y_new)) ** 2)
        
        if ss_tot < 1e-10:
            r2 = 1.0
        else:
            r2 = 1.0 - ss_res / ss_tot
        
        r2 = np.clip(r2, 0.0, 1.0)
        NS = 1.0 - r2
        
        # Combine: weight = SS * NS^power
        weight = SS * (NS ** self.novelty_power)
        
        return np.clip(weight, 0.0, 1.0)


class UncertaintyBasedWeight(WeightStrategy):
    """
    Uncertainty-based weight: Weight based on model's uncertainty on new data.
    
    Uses predictive standard deviation as a measure of uncertainty.
    High uncertainty → higher weight for new data.
    
    Useful for:
    - Active learning
    - Exploring uncertain regions
    - Adaptive sampling in sparse areas
    """
    
    def __init__(self, scale: float = 1.0):
        """
        Args:
            scale: Scaling factor for uncertainty (default: 1.0)
                   - Controls sensitivity to uncertainty
        """
        self.scale = scale
    
    def calculate(self,
                  X_old: np.ndarray,
                  y_old: np.ndarray,
                  X_new: np.ndarray,
                  y_new: np.ndarray,
                  predict_fn: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
                  **kwargs) -> float:
        """Calculate weight based on predictive uncertainty."""
        if predict_fn is None:
            raise ValueError("UncertaintyBasedWeight requires predict_fn")
        
        try:
            from Surrogates import to_numpy
        except ImportError:
            def to_numpy(arr):
                if isinstance(arr, np.ndarray):
                    return arr
                elif isinstance(arr, jnp.ndarray):
                    return np.asarray(arr)
                else:
                    return np.asarray(arr)
        
        # Predict on new data
        _, std_pred = predict_fn(X_new)
        std_pred_np = to_numpy(std_pred)
        
        # Average uncertainty
        avg_uncertainty = np.mean(std_pred_np)
        
        # Normalize by output scale (use std of y_new as reference)
        y_scale = np.std(y_new) if np.std(y_new) > 1e-10 else 1.0
        normalized_uncertainty = avg_uncertainty / y_scale
        
        # Scale and clip
        weight = normalized_uncertainty * self.scale
        return np.clip(weight, 0.0, 1.0)


# =======================================================
# Template for Custom Weight Strategies
# =======================================================

class CustomWeightTemplate(WeightStrategy):
    """
    Template for creating custom weight strategies.
    
    Copy this class and implement your own logic in the calculate() method.
    
    Example:
        class MyCustomWeight(WeightStrategy):
            def __init__(self, param1=1.0, param2=0.5):
                self.param1 = param1
                self.param2 = param2
            
            def calculate(self, X_old, y_old, X_new, y_new, predict_fn=None, **kwargs):
                # Your custom logic here
                N_old = X_old.shape[0]
                N_new = X_new.shape[0]
                
                # Example: Custom formula
                weight = self.param1 * (N_new / (N_old + N_new))
                
                return np.clip(weight, 0.0, 1.0)
    """
    
    def __init__(self, **params):
        """
        Args:
            **params: Strategy-specific parameters
        """
        self.params = params
    
    def calculate(self,
                  X_old: np.ndarray,
                  y_old: np.ndarray,
                  X_new: np.ndarray,
                  y_new: np.ndarray,
                  predict_fn: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
                  **kwargs) -> float:
        """
        Implement your custom weight calculation logic here.
        
        Returns:
            weight: Float in [0, 1]
        """
        raise NotImplementedError("Implement your custom weight calculation logic")


# =======================================================
# Convenience Functions
# =======================================================

def get_default_strategy() -> WeightStrategy:
    """
    Get the default weight strategy used in pySMC.
    
    Returns:
        SizeNoveltyWeight with novelty_power=0.5
    """
    return SizeNoveltyWeight(novelty_power=0.5)


def list_all():
    """
    List all available weight strategies.
    
    Returns:
        dict: Strategy name -> class
    """
    return {
        'uniform': UniformWeight,
        'size': SizeBasedWeight,
        'novelty': NoveltyBasedWeight,
        'size_novelty': SizeNoveltyWeight,
        'uncertainty': UncertaintyBasedWeight,
    }

