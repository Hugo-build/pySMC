"""
Universal Surrogate Modeling Framework for pySMC.

This module provides a flexible adapter framework for surrogate models with support
for adaptive sampling strategies (adaptive kriging, active learning, etc.).

Architecture:
    - BaseSurrogate: Core interface for all surrogates
    - AdaptiveSurrogate: Adds adaptive sampling capabilities
    - Model-specific adapters: Wrap concrete implementations (GP, NN, etc.)
    - Acquisition functions: Strategies for selecting next sample points
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Dict, Any, Tuple, Optional, Union, Callable, Literal
from enum import Enum

import jax.numpy as jnp
import numpy as np


# =========================
# 1) Base Surrogate Interface
# =========================

class BaseSurrogate(ABC):
    """
    Abstract base class for all surrogate models.
    
    This defines the universal interface that all surrogates must implement,
    regardless of their underlying implementation (GP, Neural Network, Polynomial, etc.).
    """
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, jnp.ndarray], 
            y: Union[np.ndarray, jnp.ndarray],
            **kwargs) -> BaseSurrogate:
        """
        Fit the surrogate model to training data.
        
        Args:
            X: Training inputs (N, D)
            y: Training outputs (N,)
            **kwargs: Model-specific options (e.g., optimizer, kernel type)
        
        Returns:
            Fitted surrogate (may be self or new instance depending on mutability)
        """
        pass
    
    @abstractmethod
    def predict(self, X_star: Union[np.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict mean and uncertainty at test points.
        
        Args:
            X_star: Test inputs (M, D)
        
        Returns:
            mean: Predictive mean (M,)
            std: Predictive uncertainty/standard deviation (M,)
        """
        pass
    
    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if the model has been fitted to data."""
        pass
    
    @staticmethod
    def validate_inputs(X: Union[np.ndarray, jnp.ndarray], 
                       y: Optional[Union[np.ndarray, jnp.ndarray]] = None,
                       fit_mode: bool = True) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Validate and convert inputs to JAX arrays.
        
        Args:
            X: Input array
            y: Output array (optional, required if fit_mode=True)
            fit_mode: Whether validating for fitting (True) or prediction (False)
        
        Returns:
            X_validated: JAX array of shape (N, D)
            y_validated: JAX array of shape (N,) or None
        """
        # Convert to JAX arrays
        X = jnp.asarray(X, dtype=jnp.float32)
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(f"X must be 1D or 2D, got shape {X.shape}")
        
        if fit_mode:
            if y is None:
                raise ValueError("y is required when fit_mode=True")
            y = jnp.asarray(y, dtype=jnp.float32)
            
            # Ensure y is 1D
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.ravel()
            elif y.ndim != 1:
                raise ValueError(f"y must be 1D, got shape {y.shape}")
            
            # Check matching samples
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y must have same number of samples. "
                    f"Got X.shape={X.shape}, y.shape={y.shape}"
                )
            
            return X, y
        
        return X, None




# =========================
# 2) Acquisition Functions for Adaptive Sampling
# =========================

class AcquisitionFunction(ABC):
    """
    Base class for acquisition functions used in adaptive sampling.
    
    Acquisition functions determine where to sample next based on the
    current surrogate model's predictions and uncertainties.
    """
    
    @abstractmethod
    def evaluate(self, mean: jnp.ndarray, std: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Evaluate the acquisition function at candidate points.
        
        Args:
            mean: Predicted mean values (M,)
            std: Predicted standard deviations (M,)
            **kwargs: Additional context (e.g., current best value)
        
        Returns:
            scores: Acquisition scores (M,) - higher is better
        """
        pass


class VarianceReduction(AcquisitionFunction):
    """
    Variance-based acquisition: Select points with highest uncertainty.
    
    Useful for space-filling and global exploration.
    """
    
    def evaluate(self, mean: jnp.ndarray, std: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Return variance (std²) as acquisition score."""
        return std ** 2


class ExpectedImprovement(AcquisitionFunction):
    """
    Expected Improvement: Balance exploration and exploitation.
    
    Useful for optimization (finding minimum/maximum).
    """
    
    def __init__(self, xi: float = 0.01):
        """
        Args:
            xi: Exploration parameter (larger = more exploration)
        """
        self.xi = xi
    
    def evaluate(self, mean: jnp.ndarray, std: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Calculate expected improvement.
        
        Requires 'best_value' in kwargs.
        """
        from jax.scipy.stats import norm
        
        best_value = kwargs.get('best_value', jnp.min(mean))
        
        # Avoid division by zero
        std = jnp.maximum(std, 1e-8)
        
        # Compute Z score
        z = (best_value - mean - self.xi) / std
        
        # Expected improvement
        ei = (best_value - mean - self.xi) * norm.cdf(z) + std * norm.pdf(z)
        return jnp.maximum(ei, 0.0)


class UpperConfidenceBound(AcquisitionFunction):
    """
    Upper Confidence Bound (UCB): Optimistic exploration.
    
    Useful for optimization with controllable exploration.
    """
    
    def __init__(self, beta: float = 2.0):
        """
        Args:
            beta: Exploration parameter (larger = more exploration)
        """
        self.beta = beta
    
    def evaluate(self, mean: jnp.ndarray, std: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Return UCB score: mean + beta * std (for maximization)."""
        # For minimization, return negative
        minimize = kwargs.get('minimize', True)
        if minimize:
            return -(mean - self.beta * std)  # Lower bound
        else:
            return mean + self.beta * std  # Upper bound


# =========================
# 3) Adaptive Surrogate Interface
# =========================

class AdaptiveSurrogate(BaseSurrogate):
    """
    Extended surrogate interface with adaptive sampling capabilities.
    
    Enables:
    - Incremental updates
    - Intelligent sample selection
    - Active learning strategies
    """
    
    @abstractmethod
    def suggest_next_sample(self,
                           X_candidates: Union[np.ndarray, jnp.ndarray],
                           acquisition: Optional[AcquisitionFunction] = None,
                           n_samples: int = 1) -> jnp.ndarray:
        """
        Suggest the next point(s) to sample based on acquisition function.
        
        Args:
            X_candidates: Candidate points to evaluate (M, D)
            acquisition: Acquisition function (default: variance reduction)
            n_samples: Number of points to suggest
        
        Returns:
            suggested_points: Selected points (n_samples, D)
        """
        pass
    
    @abstractmethod
    def add_sample(self, X_new: Union[np.ndarray, jnp.ndarray],
                   y_new: Union[np.ndarray, jnp.ndarray],
                   refit: bool = True) -> AdaptiveSurrogate:
        """
        Add new sample(s) and optionally refit the model.
        
        Args:
            X_new: New input(s) (1, D) or (K, D)
            y_new: New output(s) (1,) or (K,)
            refit: Whether to refit the model after adding samples
        
        Returns:
            Updated surrogate
        """
        pass
    
    def get_training_data(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get the current training dataset.
        
        Returns:
            X_train: Training inputs (N, D)
            y_train: Training outputs (N,)
        """
        raise NotImplementedError("Subclass must implement get_training_data()")












