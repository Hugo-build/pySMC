import numpy as np
from pathlib import Path
from typing import List
import time

# =============================================================================
# Picklable test functions (must be at module level for multiprocessing)
# =============================================================================

# The Callable class for the SobolG function
class SobolG:
    """
    Picklable Sobol G-function for multiprocessing.
    f(x) = prod((abs(4.0 * x_i - 2.0) + a_i) / (1.0 + a_i))
    
    Must be defined at module level (not inside __main__) so that
    worker processes can import it when unpickling.
    
    Args:
        a: Sensitivity parameters array
        delay: Optional delay in seconds to simulate longer computation
    """
    def __init__(self, a: np.ndarray, delay: float = 0.0, verbose: bool = False):
        self.a = np.asarray(a, dtype=float)
        self.d = self.a.size
        self.delay = delay
        self.verbose = verbose
    
    def __call__(self, x: np.ndarray) -> dict:
        
        # Simulate longer computation
        if self.delay > 0:
            time.sleep(self.delay)
        if self.verbose:
            print(f"Delayed for {self.delay} seconds")
            print(f"X input: {x}")
        
        x = np.asarray(x, dtype=float)
        val = 1.0
        for i in range(self.d):
            val *= (abs(4.0 * x[i] - 2.0) + self.a[i]) / (1.0 + self.a[i])
        return {"y": float(val)}

    def get_config(self) -> dict:
        """Return config dict (serializable for JSON)."""
        return {
            "a": self.a.tolist(),  # Convert numpy array to list for JSON
            "delay": self.delay,
            "verbose": self.verbose
        }
    
    @classmethod
    def from_config(cls, config: dict) -> "SobolG":
        """Create SobolG from config dict (loaded from JSON)."""
        return cls(
            a=np.array(config["a"]),
            delay=config.get("delay", 0.0),
            verbose=config.get("verbose", False)
        )
    
    @staticmethod
    def get_manifest(config_path: str | List[str] = None) -> dict:
        """Return manifest dict for this function."""
        return {"sobolG": config_path}

# A function which takes in a INPUT_manifest.json and evaluates the SobolG function   
def sobolg_manifest_func(configs: dict, case_id: int) -> dict:
            """
            Simulation function for ParallelRunner.
            
            Reconstructs SobolG from config and evaluates with case values.
            """
            # Reconstruct SobolG from the loaded config
            sg_config = configs["sobolG"]
            sobol_func = SobolG(
                a=np.array(sg_config["a"]),
                delay=sg_config["delay"],
                verbose=sg_config["verbose"]
            )
            
            # Get input values from _case_values (injected by runner)
            case_values = configs["_case_values"]
            x = np.array([case_values["x0"], case_values["x1"], 
                          case_values["x2"], case_values["x3"]])
            
            # Evaluate
            result = sobol_func(x)
            return result