"""
This script tests the <SklearnGPSurrogate> class created based on the <BaseSurrogate>  class.
with the Sobol G-function.
It tests the save/load functionality of the surrogate.
It tests the prediction functionality of the surrogate.
It tests the fit functionality of the surrogate.
It tests the _save_native_impl functionality of the surrogate.
It tests the _load_native_impl functionality of the surrogate.
It tests the _get_state_dict functionality of the surrogate.
It tests the _set_state_dict functionality of the surrogate.

Cautions:
- The SklearnGPSurrogate class is created based on the <BaseSurrogate> class.
- Needs to have sklearn installed.
"""


# =======================================================
#                Imports
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    GaussianProcessRegressor = None
    RBF = None
    Matern = None
    C = None
    joblib = None
    
import numpy as np
from pathlib import Path
import os
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.Surrogates import BaseSurrogate, StandardScaler, SurrogatePipe
from core.DoEs import sobol_g
# =======================================================

class SklearnGPSurrogate(BaseSurrogate):
    def __init__(self, name="GP"):
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is not installed.")
        super().__init__(name)
        # Using Matern kernel which is often better for physical/continuous functions like Sobol G
        # nu=2.5 corresponds to twice differentiable functions
        self.kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=5)
        self.supports_std = True

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X, return_std=True):
        return self.model.predict(X, return_std=return_std)

    def _save_native_impl(self, path):
        joblib.dump(self.model, os.path.join(path, "model.pkl"))
    def _load_native_impl(self, path):
        self.model = joblib.load(os.path.join(path, "model.pkl"))

    # The Hard Save (Params)
    def _get_state_dict(self):
        state = {
            "theta": self.model.kernel_.theta.tolist(),
            "alpha": self.model.alpha_.tolist(),
            "X_train": self.model.X_train_.tolist()
        }
        # Save optional attributes if they exist
        if hasattr(self.model, '_y_train_mean'):
             state["_y_train_mean"] = self.model._y_train_mean.tolist() if isinstance(self.model._y_train_mean, np.ndarray) else self.model._y_train_mean
        if hasattr(self.model, '_y_train_std'):
             state["_y_train_std"] = self.model._y_train_std.tolist() if isinstance(self.model._y_train_std, np.ndarray) else self.model._y_train_std
        # Save L_ if it exists (needed for std prediction)
        if hasattr(self.model, 'L_'):
             state["L"] = self.model.L_.tolist() if isinstance(self.model.L_, np.ndarray) else self.model.L_
        return state
    # The Hard Load (Params)
    def _set_state_dict(self, s):
        self.model = GaussianProcessRegressor(kernel=self.kernel)
        self.model.kernel_ = self.model.kernel.clone_with_theta(s["theta"])
        self.model.X_train_ = np.array(s["X_train"])
        self.model.alpha_ = np.array(s["alpha"])
        
        # Restore optional attributes
        if "_y_train_mean" in s:
            self.model._y_train_mean = np.array(s["_y_train_mean"]) if isinstance(s["_y_train_mean"], list) else s["_y_train_mean"]
        else:
            # Default for normalize_y=False
            self.model._y_train_mean = np.array([0.0])
            
        if "_y_train_std" in s:
            self.model._y_train_std = np.array(s["_y_train_std"]) if isinstance(s["_y_train_std"], list) else s["_y_train_std"]
        else:
            # Default for normalize_y=False
            self.model._y_train_std = np.array([1.0])
            
        # Restore L_
        if "L" in s:
            self.model.L_ = np.array(s["L"]) if isinstance(s["L"], list) else s["L"]
        else:
            # If L_ is missing but we need it (standard GPR), we might need to recompute it or handle it.
            # However, for prediction with return_std=True, L_ is usually required by sklearn implementation.
            # If we don't have it, we can't easily restore it without re-computing Cholesky decomposition.
            # For this test, we assume it's saved.
            pass

def test_sklearn_gp_surrogate():
    print("\n" + "="*70)
    print("TEST: SklearnGPSurrogate with Sobol G-function")
    print("="*70)

    if not SKLEARN_AVAILABLE:
        print("Skipping test: sklearn not available.")
        return

    # 1. Setup Sobol G-function as ground truth
    # Dimension 2
    a = np.array([1.0, 1.0]) 
    func = sobol_g(a)
    
    # 2. Generate training data
    n_train = 20
    X_train = np.random.rand(n_train, 2)
    y_train = np.array([func(x)["y"] for x in X_train])
    
    print(f"Generated {n_train} training samples.")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # 3. Initialize and train SklearnGPSurrogate
    gp = SklearnGPSurrogate(name="SobolGP")
    print("Initialized SklearnGPSurrogate.")

    gp.fit(X_train, y_train)
    assert gp.is_fitted, "GP should be fitted after calling fit()"
    print("✓ GP fitted successfully.")

    # 4. Generate test data
    n_test = 5
    X_test = np.random.rand(n_test, 2)
    y_test_true = np.array([func(x)["y"] for x in X_test])

    # 5. Predict
    mean, std = gp.predict(X_test)
    
    print("\nPrediction results:")
    for i in range(n_test):
        print(f"  Sample {i}: True={y_test_true[i]:.4f}, Pred={mean[i]:.4f}, Std={std[i]:.4f}")
    
    assert mean.shape == (n_test,), f"Mean shape mismatch: {mean.shape}"
    assert std.shape == (n_test,), f"Std shape mismatch: {std.shape}"
    print("✓ Prediction shape correct.")

    # 6. Test Save/Load (Native mode - pickle)
    save_dir = Path("examples/temp_sklearn_gp_test")
    if save_dir.exists():
        shutil.rmtree(save_dir)
    
    print("\nTesting Save/Load (Native)...")
    gp.save(save_dir, mode='native')
    
    gp_loaded = SklearnGPSurrogate(name="SobolGP")
    gp_loaded.load(save_dir)
    
    assert gp_loaded.is_fitted, "Loaded GP should be fitted"
    mean_loaded, std_loaded = gp_loaded.predict(X_test)
    print("Prediction results for Surrogate loaded with native:")
    for i in range(n_test):
        print(f"  Sample {i}: True={y_test_true[i]:.4f}, Pred={mean_loaded[i]:.4f}, Std={std_loaded[i]:.4f}")
    
    assert np.allclose(mean, mean_loaded), "Loaded GP prediction mismatch (mean)"
    assert np.allclose(std, std_loaded), "Loaded GP prediction mismatch (std)"
    print("✓ Native Save/Load passed.")
    
    # 7. Test Save/Load (Params mode - JSON)
    print("\nTesting Save/Load (Params)...")
    # Clean up
    if save_dir.exists():
        shutil.rmtree(save_dir)
        
    gp.save(save_dir, mode='params')
    
    gp_params = SklearnGPSurrogate(name="SobolGP")
    gp_params.load(save_dir)
    
    assert gp_params.is_fitted, "Params-loaded GP should be fitted"
    mean_params, std_params = gp_params.predict(X_test)
    
    print("Prediction results for Surrogate loaded with params:")
    for i in range(n_test):
        print(f"  Sample {i}: True={y_test_true[i]:.4f}, Pred={mean_params[i]:.4f}, Std={std_params[i]:.4f}")
    # Tolerances might be needed if exact reconstruction isn't guaranteed by sklearn's internal state
    assert np.allclose(mean, mean_params, atol=1e-5), "Params-loaded GP prediction mismatch (mean)"
    assert np.allclose(std, std_params, atol=1e-5), "Params-loaded GP prediction mismatch (std)"
    print("✓ Params Save/Load passed.")

    # Cleanup
    if save_dir.exists():
        shutil.rmtree(save_dir)

    print("\n" + "#"*70)
    print("# ALL TESTS PASSED ✓")
    print("#"*70 + "\n")

def test_sklearn_gp_pipe():
    print("\n" + "="*70)
    print("TEST: SklearnGPSurrogate with Sobol G-function")
    print("="*70)   
    if not SKLEARN_AVAILABLE:
        print("Skipping test: sklearn not available.")
        return

    # 1. Setup Sobol G-function as ground truth
    # Dimension 2
    a = np.array([1.0, 1.0]) 
    func = sobol_g(a)
    
    # 2. Generate training data
    n_train = 20
    X_train = np.random.rand(n_train, 2)
    y_train = np.array([func(x)["y"] for x in X_train])
    
    print(f"Generated {n_train} training samples.")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # 3. Initialize and train SklearnGPSurrogate
    gp = SklearnGPSurrogate(name="SobolGP")
    print("Initialized SklearnGPSurrogate.")

    # Initialize scalars
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(X_train)
    y_scaler.fit(y_train.reshape(-1, 1))

    # Create SurrogatePipe
    pipe = SurrogatePipe(
        model=gp,
        x_scaler=x_scaler,
        y_scaler=y_scaler
    )
    print("Created SurrogatePipe.")

    # 4. Fit the SurrogatePipe
    pipe.fit(X_train, y_train)
    assert pipe.is_fitted, "SurrogatePipe should be fitted after calling fit()"
    print("✓ SurrogatePipe fitted successfully.")

    # 5. Generate test data
    n_test = 5
    X_test = np.random.rand(n_test, 2)
    y_test_true = np.array([func(x)["y"] for x in X_test])

    # 6. Predict using SurrogatePipe
    # Pipe handles scaling of X and inverse scaling of y/std automatically
    mean, std = pipe.predict(X_test)
    
    print("\nPipe Prediction results:")
    for i in range(n_test):
        print(f"  Sample {i}: True={y_test_true[i]:.4f}, Pred={mean[i]:.4f}, Std={std[i]:.4f}")
    
    assert mean.shape == (n_test,), f"Mean shape mismatch: {mean.shape}"
    assert std.shape == (n_test,), f"Std shape mismatch: {std.shape}"
    print("✓ Pipe prediction shape correct.")

    # 7. Test Save/Load
    print("\nTesting SurrogatePipe Save/Load...")
    pipe_save_dir = Path("examples/temp_sklearn_pipe_test")
    if pipe_save_dir.exists():
        shutil.rmtree(pipe_save_dir)
    
    pipe.save(pipe_save_dir)
    
    # When loading, we need to provide the model instance for SklearnGPSurrogate
    # because SurrogatePipe might not know how to instantiate it or its dependencies.
    new_model = SklearnGPSurrogate(name="SobolGP")
    loaded_pipe = SurrogatePipe.load(pipe_save_dir, model=new_model)
    
    assert loaded_pipe.is_fitted, "Loaded pipe should be fitted."
    
    mean_loaded, std_loaded = loaded_pipe.predict(X_test)
    
    assert np.allclose(mean, mean_loaded), "Loaded pipe prediction mismatch (mean)"
    assert np.allclose(std, std_loaded), "Loaded pipe prediction mismatch (std)"
    print("✓ SurrogatePipe Save/Load passed.")

    # Cleanup
    if pipe_save_dir.exists():
        shutil.rmtree(pipe_save_dir)

        
    print("\n" + "#"*70)
    print("# TEST PIPE PASSED ✓")
    print("#"*70 + "\n")

if __name__ == "__main__":
    test_sklearn_gp_surrogate()
    test_sklearn_gp_pipe()
