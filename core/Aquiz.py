from __future__ import annotations
from abc import ABC, abstractmethod
import jax.numpy as jnp


# =======================================================
#  Acquisition Functions for Adaptive Sampling
# =======================================================

class AcquizFunc(ABC):
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


class VarianceMin(AcquizFunc):
    """
    Variance-based acquisition: Select points with highest uncertainty.
    
    Useful for space-filling and global exploration.
    """
    
    def evaluate(self, mean: jnp.ndarray, std: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Return variance (std²) as acquisition score."""
        return std ** 2


class ExpectedImprovement(AcquizFunc):
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


class UpperConfidenceBound(AcquizFunc):
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

