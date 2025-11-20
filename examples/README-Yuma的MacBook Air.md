# Examples, Tests, and Experiments

This directory contains example scripts, test cases, and experiments that demonstrate the functionality of the pySMC library.

## Structure

### Test Scripts (`test_*.py`)
Test scripts that validate specific functionality:
- **test_morris_simple.py**: Morris simple function test with JAX Gaussian Process
- **test_sobol_simple.py**: Sobol G-function test with JAX GP
- **test_sobol_simple_sklearn.py**: Sobol G-function test with scikit-learn GP
- **test_sobolG_verbose.py**: Verbose Sobol G-function test with kernel comparisons
- **test_surrogate_framework.py**: Comprehensive test of the Surrogate framework (SurrogatePipe, SurrogatePool, adaptive learning utilities)
- **test_parametric_variables.py**: Tests for ParametricVariable and ConfigMapper
- **test_variable_targets.py**: Tests for Variable.targets functionality
- **test_fe_direct.py**: Direct test of FE detection functions
- **test_fe_server.py**: Test script for the FE MCP server

### Example Scripts (`example_*.py`)
Example scripts demonstrating specific features:
- **example_scaled_gp.py**: Demonstrates ScaledGaussianProcess wrapper for automatic data scaling

### Experiment Scripts (`exp_*.py`)
Full experimental workflows and case studies:
- **exp_40barTruss.py**: 40-bar truss finite element analysis with parametric study and surrogate modeling
- **exp_dualOsicllator_degrade.py**: Dual oscillator system with degradation simulation

### Data Files
- **test_data.json**: Test data for various scripts

## Running Scripts

All scripts in this folder can be run from either:
1. **The project root directory**:
   ```bash
   python examples/test_morris_simple.py
   ```

2. **From within the examples directory**:
   ```bash
   cd examples
   python test_morris_simple.py
   ```

Each Python script includes path adjustments to ensure imports work correctly from both locations.

## Requirements

Scripts in this folder may require additional dependencies beyond the core pySMC requirements. See the main project's `pyproject.toml` for a complete list of dependencies.

