"""
Multiprocessing runner for batch simulations with manifest-based config injection.

Provides utilities to:
1. Run simulations in parallel using multiprocessing
2. Inject values from VariableSet before distributing to workers
3. Support SLURM and local execution environments
4. Handle config file I/O safely (read once, pass to workers)

Example:
    >>> from pySMC.io.runner import ParallelRunner
    >>> 
    >>> def my_simulation(configs: dict, case_id: int) -> dict:
    ...     # Your simulation logic here
    ...     return {"max_disp": 1.5, "max_force": 100.0}
    >>> 
    >>> runner = ParallelRunner(
    ...     manifest_path="INPUT_manifest.json",
    ...     var_set=var_set,
    ...     simu_func=my_simulation,
    ... )
    >>> results = runner.run(cases)
"""

from __future__ import annotations
import os
import sys
import copy
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# Support both package import and direct script execution
try:
    from .manifest import load_configs_from_manifest, inject_from_varset
    from ..core.Variables import Variable, VariableSet
except ImportError:
    # Running directly - add pySMC parent to path and import directly
    _pysmc_root = Path(__file__).parent.parent
    sys.path.insert(0, str(_pysmc_root.parent))  # Add MC_fishCageArray to path
    from pySMC.io.manifest import load_configs_from_manifest, inject_from_varset
    from pySMC.core.Variables import Variable, VariableSet


def get_n_workers(use_fraction: float = 0.85) -> int:
    """
    Get number of worker processes, respecting SLURM allocation.
    
    Args:
        use_fraction: Fraction of available CPUs to use (default 0.85)
        
    Returns:
        Number of workers to use
    """
    # Check SLURM environment first
    cpus_str = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus_str is None:
        cpus = os.cpu_count() or 1
    else:
        cpus = int(cpus_str)
    
    n = max(1, int(cpus * use_fraction))
    return n

# =============================================================================
# Simple function runner (no manifest, for pure functions like sobol_g)
# =============================================================================

def _worker_eval_func(args: Tuple) -> Tuple[int, Dict[str, Any]]:
    """Worker for SimpleFunctionRunner."""
    case_id, x, eval_func = args
    try:
        result = eval_func(x)
        return (case_id, {"success": True, "data": result, "x": x})
    except Exception as e:
        return (case_id, {"success": False, "error": str(e), "x": x})


class SimpleFunctionRunner:
    """
    Simple parallel runner for pure functions (no manifest needed).
    Useful for testing with analytical functions like sobol_g.

    """
    
    def __init__(
        self,
        eval_func: Callable[[np.ndarray], Dict[str, Any]],
        n_workers: int = None,
        verbose: bool = True,
    ):
        """
        Initialize simple function runner.
        
        Args:
            eval_func: Function with signature f(x) -> dict
            n_workers: Number of workers (None = auto)
            verbose: Print progress
        """
        self.eval_func = eval_func
        self.n_workers = n_workers or get_n_workers()
        self.verbose = verbose
    
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[SimpleFunctionRunner] {msg}")
    
    def run(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Evaluate function for all input samples.
        
        Args:
            X: Input array of shape (n_samples, n_dims)
            
        Returns:
            List of result dicts
        """
        X = np.asarray(X)
        n_samples = len(X)
        
        self._log(f"Running {n_samples} evaluations with {self.n_workers} workers")
        
        # ---- Prepare work items ----
        work_items = [(i, X[i], self.eval_func) for i in range(n_samples)]
        
        # ---- Run evaluations -------
        start_time = datetime.now()
        
        with mp.Pool(processes=self.n_workers) as pool:
            raw_results = pool.map(_worker_eval_func, work_items)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        self._log(f"Completed in {elapsed:.2f}s ({n_samples/elapsed:.1f} evals/sec)")
        
        # Sort by case_id
        raw_results.sort(key=lambda x: x[0])
        """ # Workers might return in random order:
            raw_results = [
                (2, {"success": True, "data": {"y": 0.5}}),
                (0, {"success": True, "data": {"y": 0.8}}),
                (1, {"success": True, "data": {"y": 0.3}}),
            ]

            # After sort:
            raw_results = [
                (0, {"success": True, "data": {"y": 0.8}}),  # case 0
                (1, {"success": True, "data": {"y": 0.3}}),  # case 1
                (2, {"success": True, "data": {"y": 0.5}}),  # case 2
            ]
        """
        results = [r[1] for r in raw_results]
        
        n_success = sum(1 for r in results if r.get("success", False))
        self._log(f"Results: {n_success}/{n_samples} successful")
        
        return results
    
    def run_and_extract(self, X: np.ndarray, key: str = "y") -> np.ndarray:
        """
        Run and extract a single output key as array.
        
        Args:
            X: Input array
            key: Key to extract from result["data"]
            
        Returns:
            Array of extracted values
        """
        results = self.run(X)
        return np.array([
            r["data"][key] if r["success"] else np.nan 
            for r in results
        ])



@dataclass
class RunnerConfig:
    """Configuration for parallel runner."""
    n_workers: Optional[int] = None  # None = auto-detect
    use_fraction: float = 0.85       # Fraction of CPUs if auto-detect
    verbose: bool = True
    save_results: bool = True
    output_dir: Path = field(default_factory=lambda: Path("."))
    output_prefix: str = "results"


# =============================================================================
# Worker function - runs outside the class to be picklable
# =============================================================================

def _worker_run_case(args: Tuple) -> Tuple[int, Dict[str, Any]]:
    """
    Worker function that runs a single simulation case.
    
    Args:
        args: Tuple of (case_id, injected_configs, simu_func)
        
    Returns:
        Tuple of (case_id, result_dict)
    """
    case_id, injected_configs, simu_func = args
    
    try:
        result = simu_func(injected_configs, case_id)
        return (case_id, {"success": True, "data": result})
    except Exception as e:
        return (case_id, {"success": False, "error": str(e)})


class ParallelRunner:
    """
    Parallel simulation runner with manifest-based config injection.
    
    The runner:
    1. Loads configs from manifest ONCE (thread-safe)
    2. Pre-injects all cases BEFORE distributing to workers
    3. Passes injected configs to worker processes
    4. Collects and saves results
    
    This design avoids file I/O race conditions by doing all config
    reading/injection in the main process.

    """
    
    def __init__(
        self,
        manifest_path: Path | str,
        var_set: VariableSet,
        simu_func: Callable[[Dict[str, Any], int], Dict[str, Any]],
        config: RunnerConfig = None,
    ):
        """
        Initialize the parallel runner.
        
        Args:
            manifest_path: Path to INPUT_manifest.json
            var_set: VariableSet defining variables and injection targets
            simu_func: Simulation function with signature:
                       simu_func(configs: dict, case_id: int) -> dict
            config: Optional RunnerConfig for customization
        """
        self.manifest_path = Path(manifest_path)
        self.var_set = var_set
        self.simu_func = simu_func
        self.config = config or RunnerConfig()
        
        # Load configs once in main process
        self._template_configs = load_configs_from_manifest(manifest_path)
        
        # Determine number of workers
        if self.config.n_workers is None:
            self.n_workers = get_n_workers(self.config.use_fraction)
        else:
            self.n_workers = self.config.n_workers
    
    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.config.verbose:
            print(f"[Runner] {msg}")
    
    def _prepare_cases(
        self,
        cases: List[List[float]] | np.ndarray,
        var_names: List[str] = None,
    ) -> List[Tuple[int, Dict[str, Any], Callable]]:
        """
        Prepare all cases by injecting values into configs.
        
        This is done in the main process to avoid file I/O in workers.
        
        Returns:
            List of (case_id, injected_configs, simu_func) tuples
        """
        if var_names is None:
            var_names = [v.name for v in self.var_set.variables]
        
        # Get workspace path from manifest location
        workspace_path = str(self.manifest_path.parent)
        
        prepared = []
        for i, case in enumerate(cases):
            # Build values dict
            values = dict(zip(var_names, case))
            
            # Inject into a fresh copy of template
            injected = inject_from_varset(self._template_configs, self.var_set, values)
            
            # Store metadata in configs for reference
            injected["_case_values"] = values
            injected["_case_id"] = i
            injected["_workspace_path"] = workspace_path  # For DIY configs & saving
            
            prepared.append((i, injected, self.simu_func))
        
        return prepared
    
    def run(
        self,
        cases: List[List[float]] | np.ndarray,
        var_names: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run simulations for all cases in parallel.
        
        Args:
            cases: List of cases, each case is a list of variable values
            var_names: Optional list of variable names. If None, uses var_set order.
            
        Returns:
            List of result dicts, ordered by case_id
        """
        cases = np.asarray(cases)
        n_cases = len(cases)
        
        self._log(f"Starting {n_cases} cases with {self.n_workers} workers")
        
        # Prepare all cases (inject values) in main process
        self._log("Preparing cases (injecting values)...")
        prepared = self._prepare_cases(cases, var_names)
        
        # Run in parallel
        self._log("Running simulations...")
        start_time = datetime.now()
        
        with mp.Pool(processes=self.n_workers) as pool:
            raw_results = pool.map(_worker_run_case, prepared)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        self._log(f"Completed in {elapsed:.1f}s")
        
        # Sort results by case_id and extract
        raw_results.sort(key=lambda x: x[0])
        results = [r[1] for r in raw_results]
        
        # Count successes/failures
        n_success = sum(1 for r in results if r.get("success", False))
        n_failed = n_cases - n_success
        
        self._log(f"Results: {n_success} success, {n_failed} failed")
        
        # Save results if configured
        if self.config.save_results:
            self._save_results(results, cases, var_names, elapsed)
        
        return results
    
    def _save_results(
        self,
        results: List[Dict[str, Any]],
        cases: np.ndarray,
        var_names: List[str],
        elapsed_time: float,
    ) -> None:
        """Save results to JSON file."""
        import json
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON file
        out_file = output_dir / f"{self.config.output_prefix}_{timestamp}.json"
        
        # Build output data structure
        var_names_list = var_names or [v.name for v in self.var_set.variables]
        
        output_data = {
            "timestamp": timestamp,
            "elapsed_time": elapsed_time,
            "n_cases": len(cases),
            "n_success": sum(1 for r in results if r.get("success", False)),
            "var_names": var_names_list,
            "cases": cases.tolist(),  # Convert numpy array to list for JSON
            "results": results,  # Already a list of dicts
        }
        
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        
        self._log(f"Saved results to {out_file}")
    
    def run_single(
        self,
        case: List[float],
        var_names: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a single case (useful for testing).
        
        Args:
            case: List of variable values
            var_names: Optional variable names
            
        Returns:
            Result dict
        """
        if var_names is None:
            var_names = [v.name for v in self.var_set.variables]
        
        values = dict(zip(var_names, case))
        injected = inject_from_varset(self._template_configs, self.var_set, values)
        injected["_case_values"] = values
        injected["_case_id"] = 0
        injected["_workspace_path"] = str(self.manifest_path.parent)
        
        return self.simu_func(injected, 0)


# =============================================================================
# SLURM job distribution helper
# =============================================================================

class SlurmJobRunner(ParallelRunner):
    """
    Extended runner for SLURM job arrays.
    
    Splits cases across multiple SLURM jobs, each job runs a subset
    of cases using local multiprocessing.
    
    Example (in SLURM script):
        >>> runner = SlurmJobRunner(
        ...     manifest_path="INPUT_manifest.json",
        ...     var_set=var_set,
        ...     simu_func=simulate,
        ...     job_id=int(os.environ["SLURM_ARRAY_TASK_ID"]),
        ...     world_jobs=int(os.environ["SLURM_ARRAY_TASK_COUNT"]),
        ... )
        >>> runner.run(all_cases)  # Only runs this job's subset
    """
    
    def __init__(
        self,
        manifest_path: Path | str,
        var_set: VariableSet,
        simu_func: Callable[[Dict[str, Any], int], Dict[str, Any]],
        job_id: int,
        world_jobs: int,
        config: RunnerConfig = None,
    ):
        super().__init__(manifest_path, var_set, simu_func, config)
        self.job_id = job_id
        self.world_jobs = world_jobs
    
    def run(
        self,
        cases: List[List[float]] | np.ndarray,
        var_names: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run only this job's subset of cases.
        
        Cases are split using strided indexing:
        job 0 gets cases [0, world_jobs, 2*world_jobs, ...]
        job 1 gets cases [1, world_jobs+1, 2*world_jobs+1, ...]
        """
        cases = np.asarray(cases)
        n_total = len(cases)
        
        # Get indices for this job (strided split)
        job_indices = np.arange(self.job_id, n_total, self.world_jobs, dtype=int)
        job_cases = cases[job_indices]
        
        self._log(f"Job {self.job_id}/{self.world_jobs}: {len(job_indices)} of {n_total} cases")
        
        # Update output prefix to include job_id
        self.config.output_prefix = f"{self.config.output_prefix}_job{self.job_id}"
        
        # Run this job's cases
        return super().run(job_cases, var_names)


# =============================================================================
# Convenience function for simple usage
# =============================================================================

def run_parallel(
    manifest_path: Path | str,
    var_set: VariableSet,
    simu_func: Callable[[Dict[str, Any], int], Dict[str, Any]],
    cases: List[List[float]] | np.ndarray,
    var_names: List[str] = None,
    n_workers: int = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convenience function to run parallel simulations.
    
    Args:
        manifest_path: Path to manifest JSON
        var_set: VariableSet for injection
        simu_func: Simulation function(configs, case_id) -> result
        cases: List of cases (each case is list of values)
        var_names: Optional variable names
        n_workers: Number of workers (None = auto)
        verbose: Print progress
        
    Returns:
        List of result dicts
        
    Example:
        >>> results = run_parallel(
        ...     "INPUT_manifest.json",
        ...     var_set,
        ...     my_simulation,
        ...     cases=[[5, 11], [3, 9]],
        ...     var_names=["Hs", "Tp"],
        ... )
    """
    config = RunnerConfig(
        n_workers=n_workers,
        verbose=verbose,
        save_results=False,
    )
    runner = ParallelRunner(manifest_path, var_set, simu_func, config)
    return runner.run(cases, var_names)



# =============================================================================
# Test with sobol_g
# =============================================================================

if __name__ == "__main__":
    import time
    try:
      from .DoEs import SobolG
    except ImportError:
      from pySMC.io.DoEs import SobolG, sobolg_manifest_func
    
    # -------------------------------------------------------
    # Test 0: Compare Sequential vs Parallel with simulated delay
    # -------------------------------------------------------
    print("=" * 60)
    print("Comparing Sequential vs Parallel (with delay)")
    print("=" * 60)
    
    # ----- Parameters -----
    a = np.array([0, 1, 4.5, 9])
    n_variables = len(a)
    n_samples = 20  # Fewer samples since we're adding delay
    delay = 0.1     # 10ms delay per evaluation
    
    # ----- Create function WITH delay to simulate real computation -----
    eval_func = SobolG(a, delay=delay, verbose=True)
    
    # ----- Generate samples -----
    X_samples = np.random.rand(n_samples, n_variables)
    
    print(f"\nSettings:")
    print(f"  n_samples = {n_samples}")
    print(f"  delay = {delay}s per evaluation")
    print(f"  Expected sequential time: {n_samples * delay:.1f}s")
    
    n_workers = get_n_workers(use_fraction=0.85)
    print(f"  n_workers = {n_workers}")
    print(f"  Expected parallel time: ~{n_samples * delay / n_workers:.1f}s")
    
    # ---- Sequential ----
    print("\n--- Sequential (single thread) ---")
    start = time.time()
    y_seq = np.array([eval_func(x)["y"] for x in X_samples])
    t_seq = time.time() - start
    print(f"Time: {t_seq:.2f}s ({n_samples/t_seq:.1f} evals/sec)")
    
    # ---- Parallel ----
    print("\n--- Parallel (multiprocessing) ---")
    runner = SimpleFunctionRunner(eval_func, n_workers=n_workers, verbose=True)
    start = time.time()
    Y_par = runner.run_and_extract(X_samples, key="y")
    t_par = time.time() - start
    
    # ---- Compare ----
    print("\n--- Results ---")
    print(f"Sequential: {t_seq:.2f}s")
    print(f"Parallel:   {t_par:.2f}s")
    print(f"Speedup:    {t_seq/t_par:.1f}x (ideal: {n_workers}x)")
    
    # ----- Verify results match -----
    assert np.allclose(y_seq, Y_par), "Results don't match!"
    print("\n✓ Results match! Test passed!")


    # -------------------------------------------------------------
    # Test 1: Test ParallelRunner with Sobol G-function
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Testing ParallelRunner with Sobol G-function")
    print("=" * 60)
    
    import tempfile
    import json
    
    # Create a temporary directory with manifest and config files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # ----- 1) Create SobolG config file -----
        sobolg_config = {
            "a": [0, 1, 4.5, 9],  # Same as above test
            "delay": 0.05,       # Smaller delay for test
            "verbose": True
        }
        config_path = tmp_path / "sobolG_config.json"
        with open(config_path, "w") as f:
            json.dump(sobolg_config, f)
        
        # ----- 2) Create manifest file -----
        manifest = {
            "files": {
                "sobolG": "sobolG_config.json"
            }
        }
        manifest_path = tmp_path / "INPUT_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        # ----- 3) Create VariableSet for x0, x1, x2, x3 (4D inputs) -----
        # These variables represent the Sobol function inputs [0, 1]
        # We don't need targets since we use _case_values directly
        var_set = VariableSet(variables=[
            Variable(name="x0", kind="uniform", params={"low": 0.0, "high": 1.0}),
            Variable(name="x1", kind="uniform", params={"low": 0.0, "high": 1.0}),
            Variable(name="x2", kind="uniform", params={"low": 0.0, "high": 1.0}),
            Variable(name="x3", kind="uniform", params={"low": 0.0, "high": 1.0}),
        ])
        
        # ----- 4) Define simulation function -----
        
        
        # ----- 5) Generate samples (using same X_samples from above) -----
        n_test_samples = 20  # Fewer samples for this test
        X_test = np.random.rand(n_test_samples, 4)
        
        # ----- 6) Run with ParallelRunner -----
        print(f"\nRunning ParallelRunner with {n_test_samples} samples...")
        
        runner_config = RunnerConfig(
            n_workers=get_n_workers(),
            verbose=True,
            save_results=False,  # Don't save for test
        )
        
        runner = ParallelRunner(
            manifest_path=manifest_path,
            var_set=var_set,
            simu_func=sobolg_manifest_func,
            config=runner_config,
        )
        
        # Run with variable names matching the var_set
        results = runner.run(
            cases=X_test,
            var_names=["x0", "x1", "x2", "x3"]
        )
        
        # ----- 7) Verify results -----
        # Compare with direct evaluation
        direct_func = SobolG(a=np.array(sobolg_config["a"]), delay=0, verbose=False)
        y_direct = np.array([direct_func(x)["y"] for x in X_test])
        y_runner = np.array([r["data"]["y"] for r in results if r["success"]])
        
        print(f"\n--- Verification ---")
        print(f"Direct evaluation: {y_direct[:5]}...")
        print(f"ParallelRunner:    {y_runner[:5]}...")
        
        if np.allclose(y_direct, y_runner):
            print("\n✓ ParallelRunner results match direct evaluation!")
        else:
            print("\n✗ Results mismatch!")
            print(f"Max diff: {np.max(np.abs(y_direct - y_runner))}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

