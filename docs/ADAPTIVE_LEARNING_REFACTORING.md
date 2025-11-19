# Adaptive Learning Refactoring Summary

## Overview

The adaptive learning functionality has been refactored into a **modular, template-based system** that separates concerns and provides maximum flexibility for users.

## Key Changes

### 1. **Removed `update_weighted()` Method**

**Before:**
```python
# Old approach - monolithic method
surrogate = surrogate.update_weighted(X_new, y_new, strategy=strategy, verbose=True)
```

**After:**
```python
# New approach - flexible composition with standalone functions
weight = calculate_adaptive_weight(
    X_old, y_old, X_new, y_new,
    predict_fn=surrogate.predict,
    strategy=strategy,
    verbose=True
)

X_combined, y_combined = combine_weighted_data(
    X_old, y_old, X_new, y_new,
    weight=weight,
    random_state=42,
    verbose=True
)

surrogate = surrogate.fit(X_combined, y_combined)
```

**Rationale:**
- **More flexible**: Users can inspect weights, modify data, or implement custom workflows
- **Testable**: Each function can be tested independently
- **Composable**: Mix and match strategies and data combination methods
- **Transparent**: Clear what's happening at each step

### 2. **New Modular Architecture**

The system now consists of three independent modules:

```
core/
├── Surrogates.py          # Surrogate model adapters
├── Weighted.py            # Weight calculation strategies  
├── Aquiz.py               # Acquisition functions
└── __init__.py            # Convenience imports
```

#### **Module: `Surrogates.py`**

**Standalone Functions:**
- `calculate_adaptive_weight()`: Calculate weight using a strategy
- `combine_weighted_data()`: Combine old/new data with weighted sampling

**Classes:**
- `BaseSurrogate`: Core interface for all surrogates
- `AdaptiveSurrogate`: Extended interface with adaptive capabilities
- `SurrogateGPax`: Adapter for JAX-based GP
- `SurrogateGPsklearn`: Adapter for sklearn GP

#### **Module: `Weighted.py`** (New!)

**Base Class:**
- `WeightStrategy`: Abstract base for all weight strategies

**Built-in Strategies:**
- `UniformWeight(weight=0.5)`: Fixed weight
- `SizeBasedWeight()`: Based on data quantity (N_new / N_old)
- `NoveltyBasedWeight(power=1.0)`: Based on prediction error (1 - R²)
- `SizeNoveltyWeight(novelty_power=0.5)`: Combined approach (default)
- `UncertaintyBasedWeight(scale=1.0)`: Based on predictive uncertainty

**Utilities:**
- `get_default_strategy()`: Returns SizeNoveltyWeight
- `list_available_strategies()`: Dictionary of all strategies

## Usage Examples

### Basic Usage (Default Strategy)

```python
from core.Surrogates import SurrogateGPax, calculate_adaptive_weight, combine_weighted_data

# Train initial surrogate
surrogate = SurrogateGPax(n_dim=3)
surrogate.fit(X_train, y_train)

# Get old data
X_old, y_old = surrogate.get_training_data()

# Calculate weight (uses SizeNoveltyWeight by default)
weight = calculate_adaptive_weight(
    X_old, y_old, X_new, y_new,
    predict_fn=surrogate.predict,
    verbose=True
)

# Combine data
X_combined, y_combined = combine_weighted_data(
    X_old, y_old, X_new, y_new,
    weight=weight,
    random_state=42
)

# Retrain
surrogate = surrogate.fit(X_combined, y_combined)
```

### Custom Strategy

```python
from core.Weighted import NoveltyBasedWeight

# Use high-sensitivity novelty-based weight
strategy = NoveltyBasedWeight(power=1.5)

weight = calculate_adaptive_weight(
    X_old, y_old, X_new, y_new,
    predict_fn=surrogate.predict,
    strategy=strategy,
    verbose=True
)
```

### Strategy Comparison

```python
from core.Weighted import (
    UniformWeight,
    SizeBasedWeight,
    NoveltyBasedWeight,
    SizeNoveltyWeight
)

strategies = {
    'uniform': UniformWeight(0.5),
    'size': SizeBasedWeight(),
    'novelty': NoveltyBasedWeight(power=1.0),
    'size_novelty': SizeNoveltyWeight(novelty_power=0.5)
}

for name, strategy in strategies.items():
    weight = calculate_adaptive_weight(
        X_old, y_old, X_new, y_new,
        predict_fn=surrogate.predict,
        strategy=strategy
    )
    print(f"{name}: {weight:.4f}")
```

### Creating Custom Strategies

```python
from core.Weighted import WeightStrategy
import numpy as np

class TimeDecayWeight(WeightStrategy):
    """Weight that increases with each update cycle."""
    
    def __init__(self, decay_rate=0.1):
        self.decay_rate = decay_rate
        self.update_count = 0
    
    def calculate(self, X_old, y_old, X_new, y_new, predict_fn=None, **kwargs):
        self.update_count += 1
        weight = 1.0 - np.exp(-self.decay_rate * self.update_count)
        return np.clip(weight, 0.0, 1.0)

# Use custom strategy
strategy = TimeDecayWeight(decay_rate=0.2)
weight = calculate_adaptive_weight(..., strategy=strategy)
```

## Benefits of New Design

### 1. **Flexibility**
- Use any combination of weight calculation and data sampling
- Easily experiment with different strategies
- Implement custom workflows (e.g., weighted updates, selective sampling)

### 2. **Transparency**
- See exactly what weight is calculated
- Inspect combined data before retraining
- Understand each step of the process

### 3. **Testability**
- Each function can be unit tested independently
- Strategies can be validated in isolation
- Easier to debug issues

### 4. **Extensibility**
- Add new weight strategies without modifying existing code
- Create domain-specific strategies easily
- Template pattern encourages best practices

### 5. **Composability**
- Mix different strategies for different outputs
- Combine with custom acquisition functions
- Integrate with external sampling methods

## Migration Guide

### Old Code (with `update_weighted`)

```python
from core.Surrogates import SurrogateGPax

surrogate = SurrogateGPax(n_dim=2)
surrogate.fit(X_train, y_train)

# Update with new data
surrogate = surrogate.update_weighted(
    X_new, y_new,
    strategy=strategy,
    random_state=42,
    verbose=True
)
```

### New Code (modular approach)

```python
from core.Surrogates import SurrogateGPax, calculate_adaptive_weight, combine_weighted_data
from core.Weighted import SizeNoveltyWeight

surrogate = SurrogateGPax(n_dim=2)
surrogate.fit(X_train, y_train)

# Get old data
X_old, y_old = surrogate.get_training_data()
X_old_np = to_numpy(X_old)
y_old_np = to_numpy(y_old)
X_new_np = to_numpy(X_new)
y_new_np = to_numpy(y_new)

# Calculate weight
strategy = SizeNoveltyWeight(novelty_power=0.5)
weight = calculate_adaptive_weight(
    X_old_np, y_old_np, X_new_np, y_new_np,
    predict_fn=surrogate.predict,
    strategy=strategy,
    verbose=True
)

# Combine data
X_combined, y_combined = combine_weighted_data(
    X_old_np, y_old_np, X_new_np, y_new_np,
    weight=weight,
    random_state=42,
    verbose=True
)

# Retrain
surrogate = surrogate.fit(X_combined, y_combined)
```

## Example Script

See `test_adaptive_learning.py` for a complete working example that:
- Generates synthetic data
- Trains initial surrogate models
- Simulates system degradation
- Compares different weight strategies
- Visualizes results

## File Structure

```
pySMC/
├── core/
│   ├── Surrogates.py          # Surrogate adapters + utility functions
│   ├── Weighted.py            # Weight strategies (NEW!)
│   ├── Aquiz.py               # Acquisition functions
│   └── __init__.py            # Exports
│
├── docs/
│   ├── ADAPTIVE_LEARNING_REFACTORING.md  # This file
│   ├── ADAPTIVE_WEIGHTS_GUIDE.md         # User guide for weight strategies
│   └── SURROGATE_ADAPTER_DESIGN.md       # Original design doc
│
├── test_adaptive_learning.py  # Example usage (NEW!)
└── archive/
    └── test_dualOscillator.py # Original experiment
```

## Design Principles

### 1. **Separation of Concerns**
- Weight calculation ≠ Data combination ≠ Model fitting
- Each component has a single responsibility

### 2. **Template Pattern**
- `WeightStrategy` defines the interface
- Concrete strategies implement specific logic
- Easy to add new strategies

### 3. **Functional Core, Imperative Shell**
- Standalone functions for core logic (pure, testable)
- Classes for state management (surrogates)

### 4. **Explicit > Implicit**
- Users see and control each step
- No hidden decisions or black boxes

## Future Extensions

Potential additions to the system:

1. **New Weight Strategies:**
   - `DistanceBasedWeight`: Based on distance in input space
   - `EnsembleWeight`: Combine multiple strategies
   - `AdaptiveThresholdWeight`: Dynamic threshold based on history

2. **Data Combination Methods:**
   - `combine_data_stratified()`: Stratified sampling
   - `combine_data_importance()`: Importance-weighted sampling
   - `combine_data_clustering()`: Cluster-based selection

3. **Utilities:**
   - `evaluate_strategy()`: Benchmark strategies on test data
   - `visualize_weight_landscape()`: Plot weight vs. parameters
   - `suggest_strategy()`: Recommend strategy based on data characteristics

## References

- Original implementation: `archive/test_dualOscillator.py` (lines 763-871)
- Weight calculation logic: Inspired by adaptive kriging literature
- Design pattern: Similar to scikit-learn's strategy pattern



