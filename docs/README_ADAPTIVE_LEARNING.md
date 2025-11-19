# Adaptive Learning Test Script

## Quick Start

Run the adaptive learning experiment:

```bash
python test_adaptive_learning.py
```

This will:
1. Generate synthetic data mimicking a dual oscillator system
2. Train initial surrogate models
3. Simulate system degradation (30% parameter shift)
4. Calculate adaptive weights using 4 different strategies
5. Update models and compare performance
6. Generate 3 visualization plots in `figs/` directory

## What This Demonstrates

### Modular Design

The script shows how to use the new modular architecture:

```python
# Step 1: Calculate weight using a strategy
weight = calculate_adaptive_weight(
    X_old, y_old, X_new, y_new,
    predict_fn=model.predict,
    strategy=SizeNoveltyWeight(novelty_power=0.5),
    verbose=True
)

# Step 2: Combine data with weighted sampling
X_combined, y_combined = combine_weighted_data(
    X_old, y_old, X_new, y_new,
    weight=weight,
    random_state=42
)

# Step 3: Retrain model
model.fit(X_combined, y_combined)
```

### Strategy Comparison

The script compares 4 built-in weight strategies:

1. **Size+Novelty (default)**: `SizeNoveltyWeight(novelty_power=0.5)`
   - Balances data quantity and prediction error
   - Best for general-purpose adaptive learning

2. **Size only**: `SizeBasedWeight()`
   - Weight = N_new / N_old
   - Focuses on data quantity

3. **Novelty only**: `NoveltyBasedWeight(power=1.0)`
   - Weight = 1 - R²
   - Focuses on how novel new data is

4. **Uniform**: `UniformWeight(weight=0.5)`
   - Fixed 50/50 balance
   - Baseline for comparison

## Output Files

The script generates 3 plots in the `figs/` directory:

1. **`adaptive_weights_comparison.png`**
   - Bar chart comparing weights from different strategies
   - Shows how each strategy prioritizes new vs old data

2. **`adaptive_learning_predictions.png`**
   - Model predictions across FA1 parameter sweep
   - Compares original model vs updated models
   - Shows how each strategy affects predictions

3. **`adaptive_learning_performance.png`**
   - R² scores for all output features
   - Compares original vs updated model performance
   - Grouped by strategy for easy comparison

## Customization

### Change Degradation Rate

```python
# In test_adaptive_learning.py, line ~141
degradation_rate = 0.5  # Increase to 50% shift
```

### Try Different Strategies

```python
# Add custom strategy to comparison
from core.Weighted import UncertaintyBasedWeight

strategies['Uncertainty'] = UncertaintyBasedWeight(scale=1.5)
```

### Modify Sample Sizes

```python
# Line ~125 - Original data
X, y = generate_synthetic_data(n_samples=500, random_state=420)

# Line ~141 - New data
X_new, y_new = generate_degraded_data(n_samples=50, ...)
```

## Code Structure

```python
# 1. Data Generation
X, y = generate_synthetic_data(n_samples=200)
X_new, y_new = generate_degraded_data(n_samples=10, degradation_rate=0.3)

# 2. Initial Training
models = train_initial_models(X_train, y_train)

# 3. Weight Calculation
for strategy_name, strategy in strategies.items():
    weight = calculate_adaptive_weight(..., strategy=strategy)

# 4. Model Update
for strategy in strategies:
    X_combined, y_combined = combine_weighted_data(..., weight=weight)
    model_updated = retrain_model(X_combined, y_combined)

# 5. Visualization
plot_weight_comparison()
plot_predictions()
plot_performance()
```

## Expected Output

```
======================================================================
ADAPTIVE LEARNING EXPERIMENT
======================================================================

[1] Training Initial Models
    Training samples: 160
    Test samples: 40
    Features: ['x1_mean', 'x2_mean', 'x1_std', 'x2_std']
    x1_mean: R² = 0.9876
    x2_mean: R² = 0.9854
    x1_std: R² = 0.9732
    x2_std: R² = 0.9698

[2] Generating New Data (Degraded System)
    New samples: 10
    Degradation rate: 30.0%

[3] Calculating Adaptive Weights
    Testing different weight strategies:

    Size+Novelty (default)   : weight = 0.4237
    Size only               : weight = 0.0625
    Novelty only            : weight = 0.7845
    Uniform (50/50)         : weight = 0.5000

[4] Updating Models with Different Strategies
    ...

[5] Generating Visualizations...
    Saved: figs/adaptive_weights_comparison.png
    Saved: figs/adaptive_learning_predictions.png
    Saved: figs/adaptive_learning_performance.png

======================================================================
EXPERIMENT COMPLETE
======================================================================
```

## Next Steps

1. **Create custom strategies**: See `core/Weighted.py` for template
2. **Real data**: Replace `generate_synthetic_data()` with your simulator
3. **Multiple outputs**: Extend to handle vector-valued outputs
4. **Sequential updates**: Loop over multiple update cycles
5. **Active learning**: Combine with acquisition functions from `core/Aquiz.py`

## Related Files

- **`core/Weighted.py`**: Weight strategy implementations
- **`core/Surrogates.py`**: Surrogate model adapters and utility functions
- **`docs/ADAPTIVE_LEARNING_REFACTORING.md`**: Design documentation
- **`docs/ADAPTIVE_WEIGHTS_GUIDE.md`**: Complete user guide
- **`archive/test_dualOscillator.py`**: Original experiment (full ODE simulation)

