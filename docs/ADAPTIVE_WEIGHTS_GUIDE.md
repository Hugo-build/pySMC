# Adaptive Weight Strategies Guide

This guide explains how to use the template-based adaptive weight system in pySMC for surrogate model updates.

## Overview

When updating a surrogate model with new data, the `update_weighted()` method uses a **weight strategy** to determine how to balance old and new data. The weight represents the relative importance of new data:

- **Weight = 0.0**: Use only old data
- **Weight = 1.0**: Use only new data
- **Weight = 0.5**: Equal balance

## Quick Start

### Using the Default Strategy

```python
from core.Surrogates import SurrogateGPax

# Create and fit surrogate
surrogate = SurrogateGPax(n_dim=2)
surrogate.fit(X_initial, y_initial)

# Update with new data (uses default SizeNoveltyWeight)
surrogate = surrogate.update_weighted(X_new, y_new, verbose=True)
```

### Using a Custom Strategy

```python
from core.Surrogates import SurrogateGPax
from core.AdaptiveWeights import NoveltyBasedWeight, UniformWeight

# Create surrogate
surrogate = SurrogateGPax(n_dim=2)
surrogate.fit(X_initial, y_initial)

# Option 1: Use novelty-based weight (high sensitivity)
strategy = NoveltyBasedWeight(power=1.0)
surrogate = surrogate.update_weighted(X_new, y_new, strategy=strategy)

# Option 2: Use uniform weight (fixed 50/50 balance)
strategy = UniformWeight(weight=0.5)
surrogate = surrogate.update_weighted(X_new, y_new, strategy=strategy)
```

## Available Strategies

### 1. SizeNoveltyWeight (Default)

Balances dataset size and data novelty:
```
Weight = (N_new / N_old) * (1 - R²)^power
```

**Parameters:**
- `novelty_power`: Exponent for novelty (default: 0.5)

**Use cases:**
- General-purpose adaptive learning
- Balancing quantity and quality of new data
- Most scenarios (this is the default)

**Example:**
```python
from core.AdaptiveWeights import SizeNoveltyWeight

# Standard (default behavior)
strategy = SizeNoveltyWeight(novelty_power=0.5)

# More sensitive to novelty
strategy = SizeNoveltyWeight(novelty_power=1.0)

# Less sensitive to novelty
strategy = SizeNoveltyWeight(novelty_power=0.2)
```

### 2. UniformWeight

Always uses a fixed weight, regardless of data.

**Parameters:**
- `weight`: Fixed weight value (default: 0.5)

**Use cases:**
- Baseline comparisons
- Fixed-ratio updates
- Testing purposes

**Example:**
```python
from core.AdaptiveWeights import UniformWeight

# Equal balance
strategy = UniformWeight(weight=0.5)

# Prefer new data
strategy = UniformWeight(weight=0.8)

# Prefer old data
strategy = UniformWeight(weight=0.2)
```

### 3. SizeBasedWeight

Weight proportional to dataset size ratio:
```
Weight = N_new / N_old (clipped to [0, 1])
```

**Use cases:**
- When data quantity matters more than quality
- Simple adaptive updates
- Computational budget constraints

**Example:**
```python
from core.AdaptiveWeights import SizeBasedWeight

strategy = SizeBasedWeight()
surrogate = surrogate.update_weighted(X_new, y_new, strategy=strategy)
```

### 4. NoveltyBasedWeight

Weight based on how novel new data is (prediction error):
```
Weight = (1 - R²)^power
```

**Parameters:**
- `power`: Exponent for novelty score (default: 1.0)

**Use cases:**
- Emphasizing new information
- Adapting to distribution shifts
- Active learning scenarios

**Example:**
```python
from core.AdaptiveWeights import NoveltyBasedWeight

# Linear sensitivity
strategy = NoveltyBasedWeight(power=1.0)

# High sensitivity to novelty
strategy = NoveltyBasedWeight(power=2.0)

# Low sensitivity
strategy = NoveltyBasedWeight(power=0.5)
```

### 5. UncertaintyBasedWeight

Weight based on model's predictive uncertainty on new data.

**Parameters:**
- `scale`: Scaling factor for uncertainty (default: 1.0)

**Use cases:**
- Active learning
- Exploring uncertain regions
- Adaptive sampling in sparse areas

**Example:**
```python
from core.AdaptiveWeights import UncertaintyBasedWeight

strategy = UncertaintyBasedWeight(scale=1.0)
surrogate = surrogate.update_weighted(X_new, y_new, strategy=strategy)
```

## Creating Custom Strategies

You can create your own weight strategy by subclassing `WeightStrategy`:

### Example 1: Time-Based Weight

```python
from core.AdaptiveWeights import WeightStrategy
import numpy as np

class TimeBasedWeight(WeightStrategy):
    """Weight that decays old data over time."""
    
    def __init__(self, decay_rate=0.1):
        self.decay_rate = decay_rate
        self.update_count = 0
    
    def calculate(self, X_old, y_old, X_new, y_new, predict_fn=None, **kwargs):
        self.update_count += 1
        # Higher weight for new data as time progresses
        weight = 1.0 - np.exp(-self.decay_rate * self.update_count)
        return np.clip(weight, 0.0, 1.0)

# Usage
strategy = TimeBasedWeight(decay_rate=0.2)
surrogate = surrogate.update_weighted(X_new, y_new, strategy=strategy)
```

### Example 2: Threshold-Based Weight

```python
from core.AdaptiveWeights import WeightStrategy
import numpy as np

class ThresholdWeight(WeightStrategy):
    """Use all new data if novelty exceeds threshold."""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def calculate(self, X_old, y_old, X_new, y_new, predict_fn=None, **kwargs):
        if predict_fn is None:
            return 0.5  # Default if no prediction function
        
        # Calculate novelty
        from core.Surrogates import to_numpy
        mean_pred, _ = predict_fn(X_new)
        mean_pred_np = to_numpy(mean_pred)
        
        ss_res = np.sum((y_new - mean_pred_np) ** 2)
        ss_tot = np.sum((y_new - np.mean(y_new)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 1.0
        novelty = 1.0 - np.clip(r2, 0.0, 1.0)
        
        # Binary decision based on threshold
        return 1.0 if novelty > self.threshold else 0.0

# Usage
strategy = ThresholdWeight(threshold=0.3)
surrogate = surrogate.update_weighted(X_new, y_new, strategy=strategy)
```

### Example 3: Composite Weight

```python
from core.AdaptiveWeights import WeightStrategy, SizeBasedWeight, NoveltyBasedWeight
import numpy as np

class CompositeWeight(WeightStrategy):
    """Combine multiple weight strategies."""
    
    def __init__(self, strategies, weights):
        """
        Args:
            strategies: List of WeightStrategy instances
            weights: List of weights for each strategy (must sum to 1)
        """
        self.strategies = strategies
        self.weights = np.array(weights) / np.sum(weights)  # Normalize
    
    def calculate(self, X_old, y_old, X_new, y_new, predict_fn=None, **kwargs):
        total_weight = 0.0
        for strategy, weight in zip(self.strategies, self.weights):
            w = strategy.calculate(X_old, y_old, X_new, y_new, predict_fn, **kwargs)
            total_weight += w * weight
        return np.clip(total_weight, 0.0, 1.0)

# Usage: 70% size-based, 30% novelty-based
strategy = CompositeWeight(
    strategies=[SizeBasedWeight(), NoveltyBasedWeight(power=0.5)],
    weights=[0.7, 0.3]
)
surrogate = surrogate.update_weighted(X_new, y_new, strategy=strategy)
```

## Tips and Best Practices

### 1. Choose Strategy Based on Your Problem

- **Stable environments**: Use `UniformWeight` or `SizeBasedWeight`
- **Shifting distributions**: Use `NoveltyBasedWeight` or `SizeNoveltyWeight`
- **Active learning**: Use `UncertaintyBasedWeight`
- **General purpose**: Use `SizeNoveltyWeight` (default)

### 2. Tune Parameters

Different problems may need different sensitivity levels:

```python
# For smooth functions (low noise)
strategy = SizeNoveltyWeight(novelty_power=0.3)  # Less sensitive

# For rough functions (high noise)
strategy = SizeNoveltyWeight(novelty_power=0.8)  # More sensitive
```

### 3. Use Verbose Mode for Debugging

```python
surrogate = surrogate.update_weighted(
    X_new, y_new, 
    strategy=strategy,
    verbose=True  # Prints weight calculation details
)
```

Output:
```
Weight Strategy: SizeNoveltyWeight
Calculated weight: 0.4237
  SS (Size Score): 0.6000
  NS (Novelty Score - 1-R²): 0.4986
Sampling 15/25 old samples, 9/15 new samples
```

### 4. Combine with Proper Sampling

The `update_weighted()` method samples data proportionally to the weight:
- If weight = 0.6, approximately 60% of samples come from new data
- If weight = 0.4, approximately 40% of samples come from new data

Control randomness with `random_state`:
```python
surrogate = surrogate.update_weighted(
    X_new, y_new, 
    strategy=strategy,
    random_state=42  # Reproducible sampling
)
```

## Advanced Usage

### Strategy Comparison

```python
from core.AdaptiveWeights import (
    UniformWeight, SizeBasedWeight, NoveltyBasedWeight, SizeNoveltyWeight
)

strategies = {
    'uniform': UniformWeight(0.5),
    'size': SizeBasedWeight(),
    'novelty': NoveltyBasedWeight(power=1.0),
    'size_novelty': SizeNoveltyWeight(novelty_power=0.5)
}

results = {}
for name, strategy in strategies.items():
    surrogate_test = surrogate.fit(X_initial, y_initial)
    surrogate_test = surrogate_test.update_weighted(X_new, y_new, strategy=strategy)
    
    # Evaluate performance
    mean_pred, _ = surrogate_test.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - mean_pred)**2))
    results[name] = rmse
    print(f"{name}: RMSE = {rmse:.4f}")
```

### Sequential Updates

```python
# Simulate multiple update cycles
for i, (X_batch, y_batch) in enumerate(data_stream):
    surrogate = surrogate.update_weighted(
        X_batch, y_batch,
        strategy=strategy,
        random_state=i,
        verbose=True
    )
    
    # Optional: Evaluate after each update
    if i % 10 == 0:
        evaluate_surrogate(surrogate, X_test, y_test)
```

## See Also

- `core/AdaptiveWeights.py` - Weight strategy implementations
- `core/Surrogates.py` - Surrogate model framework
- `core/Aquiz.py` - Acquisition functions (similar pattern)
- `docs/SURROGATE_ADAPTER_DESIGN.md` - Overall design philosophy

