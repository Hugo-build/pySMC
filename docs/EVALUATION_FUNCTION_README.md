# Evaluation Function Implementation

## Overview

A general evaluation framework has been implemented for the pySMC project. This framework allows you to:

1. **Evaluate models** (FE simulations, surrogates) for multiple samples
2. **Inject variables** into model configurations automatically
3. **Loop over cases** and collect results efficiently
4. **Compute statistics** on outputs
5. **Work with both simulations and surrogate predictors**

## Key Features ✅

- ✅ **General evaluation loop** - Works with any model type
- ✅ **Variable injection** - Uses `inject_single_config` for parametric studies
- ✅ **FE simulation support** - Full static FE solver integrated
- ✅ **Surrogate ready** - Infrastructure for surrogate predictors (stub implemented)
- ✅ **Statistics computation** - Automatic mean, std, min, max, quartiles
- ✅ **Error handling** - Robust with progress tracking
- ✅ **MCP integration** - Tools for both SA and FE workflows

## New MCP Tools

### 1. FE.evaluate_samples (proj0_FE)
General evaluation function that can be called from any workflow.

```python
FE.evaluate_samples(
    subdirectory=".",
    samples_file="sa_samples.json",
    workflow_file="sa_workflow.json",
    model_config_file="fe_config.json",
    evaluator="simulation",  # or "surrogate"
    output_file="sa_results.json",
    output_key="max_displacement"
)
```

### 2. SA.evaluate (proj0_SA)
High-level evaluation tool integrated with SA workflow.

```python
SA.evaluate(
    model_type="FE_static",
    model_config_path="fe_config.json",
    output_key="max_displacement"
)
```

### 3. SA.get_results (proj0_SA)
Retrieve evaluation results with optional row limiting.

```python
SA.get_results(
    results_file="sa_results.json",
    max_rows=10  # Optional limit for large datasets
)
```

## Quick Start

### Step 1: Define Variables (if not already done)

```python
SA.create_variable(
    name="E",
    kind="uniform",
    params={"low": 200e3, "high": 220e3},
    targets=[{"doc": "FE", "path": "elements[*].E"}]
)
```

### Step 2: Generate Samples

```python
SA.generate_samples(n_samples=50, method="sobol", seed=42)
```

### Step 3: Evaluate

```python
SA.evaluate(
    model_type="FE_static",
    model_config_path="fe_config.json"
)
```

### Step 4: Get Results

```python
results = SA.get_results(max_rows=10)
print(results["statistics"])
```

## Usage Examples

### Example 1: Parametric Study with FE

```python
# Already have variables and samples from SA workflow
SA.evaluate(
    model_type="FE_static",
    model_config_path="fe_config.json",
    output_key="max_displacement"
)

# Results saved to sa_results.json automatically
results = SA.get_results()
print(f"Mean displacement: {results['statistics']['mean']:.6e}")
```

### Example 2: Using Demo Script

```bash
cd proj0_SA
python demo_evaluation.py
```

This script demonstrates:
- Variable definition with targets
- Sample generation
- FE evaluation loop
- Results analysis

### Example 3: Evaluating with Surrogate (Future)

```python
# Train surrogate first
SA.train_surrogate(model_type="GaussianProcess")

# Evaluate using surrogate (fast!)
SA.evaluate(
    model_type="FE_static",
    evaluator="surrogate",
    surrogate_config_file="surrogate_config.json"
)
```

## Architecture

### Data Flow

```
Variables (workflow.json) ──┐
                            │
Samples (samples.json) ─────┼──> Evaluator ──> Results (results.json)
                            │       │
Model Config (fe_config) ───┘       │
                                    ▼
                              Statistics
```

### Variable Injection Process

1. Load base configuration (e.g., `fe_config.json`)
2. For each sample:
   - Map sample values to variable names
   - Use `inject_single_config` to update config
   - Evaluate modified configuration
   - Collect results
3. Compute statistics
4. Save results to JSON

## Output Format

Results are saved as JSON with this structure:

```json
{
  "n_samples": 50,
  "n_vars": 2,
  "variable_names": ["E", "F_y"],
  "output_key": "max_displacement",
  "evaluator": "simulation",
  "model_type": "FE_static",
  "results": [
    {
      "sample_id": 0,
      "inputs": {"E": 210000, "F_y": 1000},
      "output": 0.00245,
      "full_result": {
        "max_displacement": 0.00245,
        "displacements": [...]
      }
    }
  ],
  "statistics": {
    "mean": 0.0025,
    "std": 0.0003,
    "min": 0.0018,
    "max": 0.0031,
    "median": 0.0024,
    "q25": 0.0022,
    "q75": 0.0027
  },
  "created": "2025-11-12T10:30:00"
}
```

## File Structure

After running evaluation, your project will have:

```
proj0_SA/
├── sa_workflow.json      # Variables and workflow state
├── sa_samples.json       # Generated input samples
├── sa_results.json       # Evaluation outputs ← NEW
├── fe_config.json        # FE model configuration
├── demo_evaluation.py    # Demonstration script ← NEW
└── docs/
    ├── EVALUATION_GUIDE.md               ← NEW
    ├── QUICK_START_EVALUATION.md         ← NEW
    └── EVALUATION_IMPLEMENTATION_SUMMARY.md ← NEW
```

## Documentation

Comprehensive documentation has been created:

1. **EVALUATION_GUIDE.md** (proj0_SA/docs/)
   - Complete reference for evaluation framework
   - Tool descriptions and examples
   - Extension guidelines
   - Performance considerations

2. **QUICK_START_EVALUATION.md** (proj0_SA/docs/)
   - 5-minute quick start
   - Common use cases
   - Troubleshooting guide
   - Key parameters reference

3. **EVALUATION_IMPLEMENTATION_SUMMARY.md** (proj0_SA/docs/)
   - Implementation details
   - Architecture description
   - Testing checklist
   - Future enhancements

4. **demo_evaluation.py** (proj0_SA/)
   - Complete working example
   - Well-commented code
   - Demonstrates entire workflow

## Extending the Framework

### Adding a New Model Type

```python
# In SA.evaluate or FE.evaluate_samples
if model_type == "my_custom_model":
    def evaluate_config(config):
        # Your custom evaluation logic
        result = my_solver(config)
        return {
            'custom_output': result,
            'other_metrics': {...}
        }
```

### Adding New Output Keys

The `output_key` parameter allows extracting different outputs:

- `"max_displacement"` - Maximum displacement magnitude
- `"displacements"` - Full displacement vector
- `"stress"` - Stress values (if computed)
- Custom keys from your model

### Implementing Surrogate Evaluation

The surrogate evaluation infrastructure is in place. To complete it:

1. Define surrogate model storage format
2. Implement surrogate loading function
3. Implement prediction function
4. Add to evaluator switch in evaluation tools

See stub implementation at:
- `proj0_FE/server.py:1097-1126`
- `proj0_SA/server.py:773-972`

## Testing

### Run Demo Script

```bash
cd proj0_SA
python demo_evaluation.py
```

This will:
1. Create variables with targets
2. Generate 50 Sobol samples
3. Evaluate all samples with FE simulation
4. Compute and display statistics
5. Save results to JSON files

### Verify Installation

Check that all files are in place:

```bash
# New tools in servers
grep -n "FE.evaluate_samples" proj0_FE/server.py
grep -n "SA.evaluate" proj0_SA/server.py
grep -n "SA.get_results" proj0_SA/server.py

# New documentation
ls proj0_SA/docs/EVALUATION*.md

# Demo script
ls proj0_SA/demo_evaluation.py
```

## Performance Tips

1. **Start small**: Test with 10-20 samples first
2. **Use Sobol sampling**: Better coverage than random
3. **Train surrogates**: For >100 samples
4. **Monitor progress**: Large evaluations take time
5. **Limit output size**: Use `output_key` to extract only what you need

## Troubleshooting

### "Samples file not found"
**Solution:** Run `SA.generate_samples()` first

### "Model config file not found"  
**Solution:** Ensure `fe_config.json` exists with nodes, elements, fixed_dofs, loads

### "Variable 'X' not found in config"
**Solution:** Check that variable targets match config structure

### "Singular matrix"
**Solution:** Check boundary conditions - model may be under-constrained

## Next Steps

After evaluation, you can:

1. **Train surrogate model** (when implemented):
   ```python
   SA.train_surrogate(model_type="GaussianProcess")
   ```

2. **Compute sensitivity indices** (when implemented):
   ```python
   SA.compute_sobol(use_surrogate=True)
   ```

3. **Visualize results**:
   - Plot input-output relationships
   - Analyze sensitivity
   - Study output distributions

4. **Use surrogate for UQ**:
   - Fast predictions for large sample sets
   - Monte Carlo with millions of samples
   - Real-time what-if analysis

## Code Quality

- ✅ No linter errors
- ✅ Comprehensive docstrings
- ✅ Type hints where applicable
- ✅ Robust error handling
- ✅ Well-tested with demo script

## Summary

The evaluation framework is now **fully functional** for FE simulations and **ready for surrogate implementation**. You can:

- ✅ Evaluate FE models for multiple samples
- ✅ Inject variables automatically
- ✅ Get comprehensive results with statistics
- ✅ Use MCP tools for workflow integration
- 🔄 Add surrogate evaluation (infrastructure ready)

All documentation and examples are in place for immediate use!

## Related Files

- `proj0_FE/server.py` - FE evaluation tools
- `proj0_SA/server.py` - SA evaluation tools  
- `proj0_SA/demo_evaluation.py` - Working example
- `proj0_SA/docs/EVALUATION_GUIDE.md` - Complete guide
- `proj0_SA/docs/QUICK_START_EVALUATION.md` - Quick reference
- `examples/exp_40barTruss.py` - Full FE example with variables
- `core/Variables.py` - Variable and injection utilities

