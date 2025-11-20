"""
Test script demonstrating the new import structure of pySMC.

This script shows different ways to import from the pySMC package now that
the __init__.py files have been properly configured.
"""

# =============================================================================
# Method 1: Import everything from the main package
# =============================================================================
print("=" * 70)
print("Method 1: Import from main pySMC package")
print("=" * 70)

try:
    # Now you can import directly from pySMC
    from pySMC import (
        # Surrogate modeling
        SurrogatePipe,
        SurrogatePool,
        StandardScaler,
        calc_upd_weight,
        combine_weighted_data,
        to_numpy,
        to_jax,
        
        # Gaussian Process
        GaussianProcess,
        RBF,
        Matern32,
        Matern52,
        optSetup,
        
        # Variables
        Variable,
        VariableSet,
        
        # Sampling
        sample_inputs,
        
        # Design of Experiments
        sobol_g,
        morris_g,
        
        # Data preprocessing
        train_test_split,
        remove_nan,
        scale_data,
        
        # Weighting strategies
        SizeNoveltyWeight,
        UniformWeight,
        
        # Acquisition functions
        VarianceMin,
        ExpectedImprovement,
        UpperConfidenceBound,
        
        # Monte Carlo
        run_monte_carlo,
        MCResult,
    )
    print("✓ Successfully imported all main components from pySMC")
    print(f"  - SurrogatePipe: {SurrogatePipe}")
    print(f"  - GaussianProcess: {GaussianProcess}")
    print(f"  - Variable: {Variable}")
except ImportError as e:
    print(f"✗ Import failed: {e}")

# =============================================================================
# Method 2: Import from core submodule
# =============================================================================
print("\n" + "=" * 70)
print("Method 2: Import from pySMC.core submodule")
print("=" * 70)

try:
    # You can also import from the core submodule
    from pySMC.core import (
        GaussianProcess,
        RBF,
        Variable,
        VariableSet,
        sample_inputs,
        SurrogatePipe,
    )
    print("✓ Successfully imported from pySMC.core")
except ImportError as e:
    print(f"✗ Import failed: {e}")

# =============================================================================
# Method 3: Import specific modules from core (old style - still works)
# =============================================================================
print("\n" + "=" * 70)
print("Method 3: Import from specific core modules (legacy)")
print("=" * 70)

try:
    # Legacy style imports still work
    from pySMC.core.GPax import GaussianProcess, RBF, Matern52
    from pySMC.core.Variables import Variable, VariableSet
    from pySMC.core.Samplers import sample_inputs
    from pySMC.core.Surrogates import SurrogatePipe
    from pySMC.core.DoEs import sobol_g
    print("✓ Successfully imported using legacy style")
except ImportError as e:
    print(f"✗ Import failed: {e}")

# =============================================================================
# Method 4: Check package metadata
# =============================================================================
print("\n" + "=" * 70)
print("Method 4: Check package metadata")
print("=" * 70)

try:
    import pySMC
    print(f"✓ Package: {pySMC.__name__}")
    print(f"✓ Version: {pySMC.__version__}")
    print(f"✓ Author: {pySMC.__author__}")
    print(f"✓ Exported symbols: {len(pySMC.__all__)} items")
    print(f"\nFirst 10 exports:")
    for item in pySMC.__all__[:10]:
        print(f"  - {item}")
    print(f"  ... and {len(pySMC.__all__) - 10} more")
except Exception as e:
    print(f"✗ Failed to read metadata: {e}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
The pySMC package now supports clean imports in three ways:

1. Direct imports from main package (RECOMMENDED):
   from pySMC import GaussianProcess, Variable, sample_inputs
   
2. Import from core submodule:
   from pySMC.core import GaussianProcess, Variable, sample_inputs
   
3. Import from specific modules (legacy, still works):
   from pySMC.core.GPax import GaussianProcess
   from pySMC.core.Variables import Variable
   
All three methods work, but Method 1 is recommended for new code.
""")
print("=" * 70)

