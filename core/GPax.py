from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Protocol, Dict, Any, Tuple, Optional, Mapping, Callable

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky, cho_solve, cho_factor, solve_triangular
import optax



# =========================
# 1) Helper functions
# =========================
def _ard_sq_dists(X1: jnp.ndarray, X2: jnp.ndarray, ls: jnp.ndarray) -> jnp.ndarray:
    """Compute squared distances between X1 and X2 using ARD (Automatic Relevance Determination)."""
    return jnp.sum((X1[:, None] - X2[None, :])**2 / ls**2, axis=2)


def _param_change_norm(params_old: Mapping[str, Any], params_new: Mapping[str, Any]) -> float:
    """
    Compute the L2 norm of parameter changes between two parameter trees.
    
    Args:
        params_old: Previous parameter tree
        params_new: Current parameter tree
    
    Returns:
        Global norm of parameter changes
    """
    def _diff_tree(p1, p2):
        return p1 - p2
    
    diff_tree = jax.tree_util.tree_map(_diff_tree, params_new, params_old)
    return float(optax.global_norm(diff_tree))


@dataclass(frozen=True)
class optSetup:
    """
    Configuration for hyperparameter optimization.
    
    Simplified to only accept optimizer names as strings.
    Supported optimizers: 'adam', 'lbfgs', 'sgd'
    
    Stopping criteria (similar to MATLAB's fitrgp):
    - tol_fun: Function tolerance - stop if |loss_new - loss_old| < tol_fun
    - tol_x: Step tolerance - stop if parameter change norm < tol_x
    - tol_grad: Gradient tolerance - stop if gradient norm < tol_grad
    - steps: Maximum number of iterations (safety limit)
    - patience: Early stopping patience - stop if no improvement for N consecutive steps
    """
    optimizer: str = 'adam'  # Optimizer name: 'adam', 'lbfgs', 'sgd'
    steps: int = 100
    lr: float = 0.01
    prior_fn: Optional[Callable[[Dict[str, Any]], jnp.ndarray]] = None
    log_every: int = 10
    verbose: bool = False

    # Stopping criteria (None means disabled)
    tol_fun: Optional[float] = 1e-6      # Function tolerance (MATLAB default: 1e-6)
    tol_x: Optional[float] = 1e-8        # Step tolerance (MATLAB default: 1e-8)
    tol_grad: Optional[float] = 1e-5     # Gradient tolerance (MATLAB default: 1e-5)
    patience: Optional[int] = None       # Early stopping patience (None = disabled)
    


def get_optimizer(optimizer_name: str, lr: float) -> optax.GradientTransformation:
    """
    Create an optimizer instance from its name.
    
    Args:
        optimizer_name: Name of the optimizer ('adam', 'lbfgs', 'sgd')
        lr: Learning rate
    
    Returns:
        Optax optimizer instance
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return optax.adam(learning_rate=lr)
    elif optimizer_name == 'lbfgs':
        return optax.lbfgs(learning_rate=lr)
    elif optimizer_name == 'sgd':
        return optax.sgd(learning_rate=lr, momentum=None)
    else:
        raise ValueError(
            f"Unknown optimizer '{optimizer_name}'. "
            f"Supported optimizers: 'adam', 'lbfgs', 'sgd'"
        )


def step_4_lbfgs(
    optimizer: optax.GradientTransformation,
    loss_fn: Callable[[Mapping[str, Any]], jnp.ndarray]
) -> Callable[[Mapping[str, Any], optax.OptState], Tuple[Mapping[str, Any], optax.OptState, jnp.ndarray]]:
    """
    Create a JIT-compiled LBFGS optimization step function.
    
    Args:
        optimizer: LBFGS optimizer instance
        loss_fn: Loss function that takes parameters and returns scalar loss
    
    Returns:
        JIT-compiled step function
    """
    @jax.jit
    def step(params: Mapping[str, Any], state: optax.OptState) -> Tuple[Mapping[str, Any], optax.OptState, jnp.ndarray]:
        """LBFGS optimization step with special handling for value_fn."""
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, state = optimizer.update(
            grads, state, params,
            value=loss,           # Current loss value
            grad=grads,           # Gradients (redundant but required by LBFGS)
            value_fn=loss_fn      # Loss function for line search
        )
        params = optax.apply_updates(params, updates)
        return params, state, loss, grads
    
    return step


def step_4_stdGrad(
    optimizer: optax.GradientTransformation,
    loss_fn: Callable[[Mapping[str, Any]], jnp.ndarray]
) -> Callable[[Mapping[str, Any], optax.OptState], Tuple[Mapping[str, Any], optax.OptState, jnp.ndarray]]:
    """
    Create a JIT-compiled standard optimization step function.
    
    Args:
        optimizer: Optimizer instance (Adam, SGD, etc.)
        loss_fn: Loss function that takes parameters and returns scalar loss
    
    Returns:
        JIT-compiled step function
    """
    @jax.jit
    def step(params: Mapping[str, Any], state: optax.OptState) -> Tuple[Mapping[str, Any], optax.OptState, jnp.ndarray]:
        """Standard optimization step for first-order methods."""
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, loss, grads
    
    return step


def get_stepFunc(
    optimizer_name: str,
    optimizer: optax.GradientTransformation,
    loss_fn: Callable[[Mapping[str, Any]], jnp.ndarray]
) -> Callable[[Mapping[str, Any], optax.OptState], Tuple[Mapping[str, Any], optax.OptState, jnp.ndarray]]:
    """
    Get the appropriate optimization step function based on optimizer type.
    
    Args:
        optimizer_name: Name of the optimizer ('adam', 'lbfgs', 'sgd')
        optimizer: Optimizer instance
        loss_fn: Loss function that takes parameters and returns scalar loss
    
    Returns:
        JIT-compiled step function appropriate for the optimizer
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'lbfgs':
        return step_4_lbfgs(optimizer, loss_fn)
    else:
        return step_4_stdGrad(optimizer, loss_fn)


# =========================
# 2) Typed callable kernels
# =========================
class Kernel(Protocol):
    """
    Callable kernel interface using Protocol for structural typing.
    
    FIXED: Added get_params_tree() method and fixed typo (paramms → params).
    """
    def __call__(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray: ...

    def with_params(self, params: Mapping[str, Any]) -> Kernel:  # FIX #5: Fixed typo
        """Create a new kernel with updated parameters."""
        ...
    
    def get_params_tree(self) -> Dict[str, Any]:  # FIX #3: Added missing method
        """Get the kernel parameters as a tree of dictionaries."""
        ...


@dataclass(frozen=True)
class RBF:
    """
    Radial Basis Function (Squared Exponential) kernel.
    
    k(x, x') = σ² exp(-||x - x'||² / (2ℓ²))
    """
    log_sf: jnp.ndarray  # Log signal standard deviation
    log_ls: jnp.ndarray  # Log length scale(s) - can be vector for ARD

    def __call__(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        sf = jnp.exp(self.log_sf)         # The signal variance
        ls = jnp.exp(self.log_ls)         # The length scale
        dists = _ard_sq_dists(X1, X2, ls) # The squared distances between X1 and X2
        return sf**2 * jnp.exp(-dists / 2) # The RBF kernel

    def with_params(self, params: Mapping[str, Any]) -> RBF:
        """Create a new kernel with updated parameters."""
        return replace(self, 
                       log_sf=params.get("log_sf", self.log_sf),
                       log_ls=params.get("log_ls", self.log_ls))
    
    def get_params_tree(self) -> Dict[str, Any]:  # FIX #3: Added to all kernels
        """Get the kernel parameters as a dictionary."""
        return {"log_sf": self.log_sf, "log_ls": self.log_ls}

@dataclass(frozen=True)
class Matern32:
    """
    Matérn 3/2 kernel - smoother than RBF but allows for some roughness.
    
    k(x, x') = σ² (1 + √3r) exp(-√3r), where r = ||x - x'|| / ℓ
    """
    log_sf: jnp.ndarray  # Log signal standard deviation
    log_ls: jnp.ndarray  # Log length scale(s)

    def __call__(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        sf = jnp.exp(self.log_sf) # The signal variance
        ls = jnp.exp(self.log_ls) # The length scale
        r = jnp.sqrt(jnp.maximum(_ard_sq_dists(X1, X2, ls), 1e-10))
        c = jnp.sqrt(3.0) 
        return sf**2 * (1.0 + c * r) * jnp.exp(-c * r)

    def with_params(self, params: Mapping[str, Any]) -> Matern32:
        """Create a new kernel with updated parameters."""
        return replace(self, 
                       log_sf=params.get("log_sf", self.log_sf),
                       log_ls=params.get("log_ls", self.log_ls))
    
    def get_params_tree(self) -> Dict[str, Any]:  # FIX #3: Added to all kernels
        """Get the kernel parameters as a dictionary."""
        return {"log_sf": self.log_sf, "log_ls": self.log_ls}

@dataclass(frozen=True)
class Matern52:
    """
    Matérn 5/2 kernel - twice differentiable, very smooth.
    
    k(x, x') = σ² (1 + √5r + 5r²/3) exp(-√5r), where r = ||x - x'|| / ℓ
    """
    log_sf: jnp.ndarray  # Log signal standard deviation
    log_ls: jnp.ndarray  # Log length scale(s)

    def __call__(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        sf = jnp.exp(self.log_sf) # The signal variance
        ls = jnp.exp(self.log_ls) # The length scale
        r = jnp.sqrt(jnp.maximum(_ard_sq_dists(X1, X2, ls), 1e-10))
        c = jnp.sqrt(5.0) 
        return sf**2 * (1.0 + c * r + 5.0/3.0 * r**2) * jnp.exp(-c * r)

    def with_params(self, params: Mapping[str, Any]) -> Matern52:
        """Create a new kernel with updated parameters."""
        return replace(self, 
                       log_sf=params.get("log_sf", self.log_sf),
                       log_ls=params.get("log_ls", self.log_ls))
    
    def get_params_tree(self) -> Dict[str, Any]:  # FIX #3: Added to all kernels
        """Get the kernel parameters as a dictionary."""
        return {"log_sf": self.log_sf, "log_ls": self.log_ls}



# =========================
# 3) Gaussian Process
# =========================

@dataclass(frozen=True)
class GaussianProcess:
    """
    Gaussian process surrogate model using JAX.
    
    A functional, immutable GP implementation with:
     - kernel: Callable kernel object following the Kernel protocol
     - log_sn2: Log of observation noise variance
     - jitter: Small diagonal term for numerical stability (default 1e-6)
     - X, y: Training data (stored after fitting)
    
    FIXED: This is a frozen dataclass, so all updates use replace() to create new instances.
    """
    kernel: Kernel
    log_sn2: jnp.ndarray
    jitter: float = 1e-6
    X: Optional[jnp.ndarray] = None
    y: Optional[jnp.ndarray] = None

    def get_params_tree(self) -> Dict[str, Any]:
        """
        Get all optimizable parameters as a nested dictionary.
        
        FIX #3: Now properly uses kernel.get_params_tree() which exists in all kernels.
        """
        return {
            "kernel": self.kernel.get_params_tree(),  # Now works!
            "log_sn2": self.log_sn2,
            # Note: jitter is typically not optimized
        }
    
    def set_params_tree(self, tree: Mapping[str, Any]) -> GaussianProcess:
        """
        Create a new GP with updated parameters from a tree structure.
        
        This is key for functional optimization - returns a NEW instance.
        """
        k_params = tree.get("kernel", {})
        new_kernel = self.kernel.with_params(k_params)
        
        return replace(
            self, 
            kernel=new_kernel, 
            log_sn2=tree.get("log_sn2", self.log_sn2), 
            jitter=tree.get("jitter", self.jitter)
        ) 

    # The negative log marginal likelihood is the objective function to minimize.
    def neg_lml(self, X: jnp.ndarray, y: jnp.ndarray,
                params: Optional[Mapping[str, Any]] = None,
                prior_fn: Optional[Callable[[Dict[str, Any]], jnp.ndarray]] = None) -> jnp.ndarray:
        """
        Compute negative log marginal likelihood (for maximization via minimization).
        
        The NLML = 0.5 * y^T K^{-1} y + 0.5 * log|K| + 0.5 * n * log(2π)
        
        Optionally adds prior regularization if prior_fn is provided.
        """
        if params is not None:
            gp = self.set_params_tree(params)
        else:
            gp = self
            
        K = gp.kernel(X, X)
        sn2 = jnp.exp(gp.log_sn2)
        Ky = K + (sn2 + gp.jitter) * jnp.eye(X.shape[0])

        c, lower = cho_factor(Ky, lower=True)
        alpha = cho_solve((c, lower), y)
        n = X.shape[0]
        
        # Three components of NLML
        data_fit = 0.5 * jnp.dot(y, alpha)
        logdet = jnp.sum(jnp.log(jnp.diag(c)))  # c is already from Cholesky
        constant = 0.5 * n * jnp.log(2.0 * jnp.pi)

        objective = data_fit + logdet + constant
        
        # Add prior penalty if specified (for MAP estimation)
        if prior_fn is not None:
            objective = objective - prior_fn(gp.get_params_tree())  # Subtract log prior
            
        return objective
       
    # The posterior mean and covariance are the predictions of the GP.
    def posterior(self, X: jnp.ndarray,
                  y: jnp.ndarray,
                  X_star: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the posterior mean and covariance at test points X_star.
        
        Returns:
            mean: Posterior mean (M,)
            cov: Posterior covariance (M, M)
        """
        K = self.kernel(X, X)
        Ks = self.kernel(X_star, X)
        Kss = self.kernel(X_star, X_star)
        sn2 = jnp.exp(self.log_sn2)

        Ky = K + (sn2 + self.jitter) * jnp.eye(X.shape[0])
        c, lower = cho_factor(Ky, lower=True)
        alpha = cho_solve((c, lower), y)
        v = solve_triangular(c, Ks.T, lower=True)  # Solve L v = Ks^T
        
        mean = Ks @ alpha              # Posterior mean (M,)
        cov = Kss - v.T @ v            # Posterior covariance (M, M)
        return mean, cov
    
    def predict(self, X_star: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict at test points using stored training data.
        
        Returns:
            mean: Predictive mean (M,)
            std: Predictive standard deviation (M,)
        """
        if self.X is None or self.y is None:
            raise ValueError(
                "Gaussian process must be fitted before predicting! "
                "Call .fit(X, y, optimize=...) first."
            )
        mean, cov = self.posterior(self.X, self.y, X_star)
        var = jnp.clip(jnp.diag(cov), 0.0, jnp.inf)  # Clip numerical negatives
        return mean, jnp.sqrt(var)
    
    def fit(self, X: jnp.ndarray, y: jnp.ndarray,
            opt_config: Optional[optSetup] = None) -> GaussianProcess:
        """
        Fit the GP to training data, optionally optimizing hyperparameters.
        
        FIX #2: Use replace() to store X, y (frozen dataclass!)
        FIX #1: Fixed indentation - return AFTER optimization loop, not inside!
        FIX #4: Pass prior_fn to neg_lml during optimization.
        FIX #8: Always return a GP instance.
        
        Args:
            X: Training inputs (N, D)
            y: Training outputs (N,)
            optimize: Optional optimization config. If None, just stores data.
        
        Returns:
            New GaussianProcess instance with optimized hyperparameters and stored data.
        """
        # FIX #2: Can't do self.X = X on frozen dataclass! Use replace()
        gp = replace(self, X=X, y=y)
        
        if opt_config is None:
            return gp  # FIX #8: Just store data, no optimization
        
        # Get optimizer instance from name
        optimizer = get_optimizer(opt_config.optimizer, opt_config.lr)
        
        params = gp.get_params_tree()
        
        # Define the objective function
        def loss_fn(p):
            return self.neg_lml(X, y, p, prior_fn=opt_config.prior_fn)
        
        # Get appropriate step function for the optimizer
        step_fn = get_stepFunc(opt_config.optimizer, optimizer, loss_fn)
        
        # Optimization loop with convergence checking
        state = optimizer.init(params)
        prev_loss = float('inf')
        prev_params = params
        best_loss = float('inf')
        best_params = params
        no_improvement_count = 0
        converged = False
        convergence_reason = None
        
        print("Starting optimization...")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        for i in range(1, opt_config.steps + 1):
            params, state, loss, grads = step_fn(params, state)
            
            # Track best loss for early stopping
            current_loss = float(loss)
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Check convergence criteria
            grad_norm = optax.global_norm(grads)
            
            # Function tolerance check (Tolerance for the function value)
            if opt_config.tol_fun is not None and i > 1 and abs(current_loss - prev_loss) < opt_config.tol_fun:
                converged = True
                convergence_reason = f"Function tolerance ({opt_config.tol_fun:.2e})"
                break
            
            # Parameter change tolerance check (Tolerance for the parameter change)
            if opt_config.tol_x is not None and i > 1:
                param_change = _param_change_norm(prev_params, params)
                if param_change < opt_config.tol_x:
                    converged = True
                    convergence_reason = f"Step tolerance ({opt_config.tol_x:.2e})"
                    break
            
            # Gradient tolerance check (Tolerance for the gradient)
            if opt_config.tol_grad is not None and grad_norm < opt_config.tol_grad:
                converged = True
                convergence_reason = f"Gradient tolerance ({opt_config.tol_grad:.2e})"
                break
            
            # Early stopping patience check (Early stopping patience)
            if opt_config.patience is not None and no_improvement_count >= opt_config.patience:
                converged = True
                convergence_reason = f"Early stopping (patience={opt_config.patience})"
                # Use best parameters found so far
                params = best_params
                break
            
            # Logging
            if opt_config.verbose and i % opt_config.log_every == 0:
                param_change = _param_change_norm(prev_params, params) if i > 1 else float('inf')
                print(f"Step {i:4d}: Loss = {current_loss:.6f}, Grad Norm = {grad_norm:.6f}, "
                      f"Param Change = {param_change:.6f}")
            
            prev_loss = current_loss
            prev_params = params
        
        # Log final convergence status
        if opt_config.verbose:
            if converged:
                print(f"Optimization converged after {i} steps: {convergence_reason}")
                print(" Maybe it is the best parameters found so far")
            else:
                print(f"Optimization reached maximum iterations ({opt_config.steps})")
        
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Optimization finished")
        # Update both parameters AND store data
        return replace(gp.set_params_tree(params), X=X, y=y) 
           
            




# ===========================
# 4) Example usage
# ===========================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datetime import datetime
    import os
    
    key = jax.random.key(42)
    
    # Generate synthetic data
    N = 50
    X_train = jnp.linspace(0.0, 6.0, N).reshape(-1, 1)
    myFunc = lambda x: jnp.sin(2*jnp.pi*x)
    f_true = myFunc(X_train).squeeze()
    noise_std = 0.15
    y_train = f_true + noise_std * jax.random.normal(key, (N,))
    
    # Initialize kernel with reasonable hyperparameters
    D = X_train.shape[1]
    kernel = RBF(
        log_sf=jnp.log(jnp.std(y_train) + 1e-6),
        log_ls=jnp.log(jnp.ones((D,)) * ((X_train.max() - X_train.min()) / 4.0)),
    )
    
    # Create GP with initial hyperparameters
    gp = GaussianProcess(
        kernel=kernel, 
        log_sn2=jnp.log(jnp.array(noise_std**2)), 
        jitter=1e-6
    )
    
    # Fit WITHOUT optimization (just store data with initial hyperparameters)
    print("Fitting GP WITHOUT hyperparameter optimization...")
    gp_no_opt = gp.fit(X_train, y_train, opt_config=None)
    
    # Setup optimization
    # Available optimizers (just use the name as a string):
    # 1. 'adam' (default, reliable) - recommended for most cases
    # 2. 'lbfgs' (faster convergence) - good for smooth objectives
    # 3. 'sgd' (simple baseline) - basic gradient descent with momentum
    
    opt_config = optSetup(
        optimizer='adam',  # Simply specify the optimizer name
        steps=100,
        lr=0.001,  # Learning rate for the optimizer
        log_every=10,
        verbose=True
    )
    
    # If you want to try LBFGS:
    # opt_config = optSetup(
    #     optimizer='lbfgs',
    #     steps=50,
    #     lr=1.0,
    #     log_every=10,
    #     verbose=True
    # )
    
    # Fit WITH optimization (optimizes hyperparameters and stores data)
    print("\nFitting GP WITH hyperparameter optimization...")
    gp_optimized = gp.fit(X_train, y_train, opt_config=opt_config)
    
    # Make predictions on test points
    X_test = jnp.linspace(-1.0, 7.0, 200).reshape(-1, 1)
    mean_no_opt, std_no_opt = gp_no_opt.predict(X_test)
    mean_opt, std_opt = gp_optimized.predict(X_test)
    
    # Compare hyperparameters
    print("\n" + "="*60)
    print("HYPERPARAMETER COMPARISON")
    print("="*60)
    
    params_init = gp_no_opt.get_params_tree()
    params_opt = gp_optimized.get_params_tree()
    
    def print_params(params, label):
        """Helper to print parameters nicely."""
        log_sf = float(params['kernel']['log_sf'])
        log_ls = params['kernel']['log_ls']
        log_sn2 = float(params['log_sn2'])
        
        print(f"\n{label}:")
        print(f"  Signal variance:  log_sf={log_sf:.3f}  →  sf={jnp.exp(log_sf):.3f}")
        
        if log_ls.ndim == 0 or (log_ls.ndim == 1 and log_ls.shape[0] == 1):
            log_ls_val = float(log_ls.squeeze())
            print(f"  Length scale:     log_ls={log_ls_val:.3f}  →  ls={jnp.exp(log_ls_val):.3f}")
        else:
            print(f"  Length scale:     log_ls={log_ls}  →  ls={jnp.exp(log_ls)}")
        
        print(f"  Noise variance:   log_sn2={log_sn2:.3f}  →  sn={jnp.sqrt(jnp.exp(log_sn2)):.3f}")
    
    print_params(params_init, "Initial (no optimization)")
    print_params(params_opt, "Optimized")
    
    # Compute negative log marginal likelihoods for comparison
    nlml_no_opt = gp_no_opt.neg_lml(X_train, y_train)
    nlml_opt = gp_optimized.neg_lml(X_train, y_train)
    
    print(f"\nNegative Log Marginal Likelihood:")
    print(f"  Without optimization: {nlml_no_opt:.2f}")
    print(f"  With optimization:    {nlml_opt:.2f}")
    print(f"  Improvement:          {nlml_no_opt - nlml_opt:.2f} (lower is better)")
    print("="*60)
    


    # ------------------------------------------------------------
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: No optimization
    ax1 = axes[0]
    ax1.plot(X_train, y_train, 'ko', label='Training data', markersize=5, zorder=3)
    ax1.plot(X_test, mean_no_opt, 'b-', label='GP mean', linewidth=2, zorder=2)
    ax1.fill_between(
        X_test.squeeze(), 
        (mean_no_opt - 2*std_no_opt).squeeze(), 
        (mean_no_opt + 2*std_no_opt).squeeze(), 
        alpha=0.3, 
        label='95% confidence',
        zorder=1
    )
    ax1.plot(X_test, myFunc(X_test), 'r--', label='True function', linewidth=1.5, zorder=2)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Without Hyperparameter Optimization', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f'NLML = {nlml_no_opt:.2f}', 
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right plot: With optimization
    ax2 = axes[1]
    ax2.plot(X_train, y_train, 'ko', label='Training data', markersize=5, zorder=3)
    ax2.plot(X_test, mean_opt, 'b-', label='GP mean', linewidth=2, zorder=2)
    ax2.fill_between(
        X_test.squeeze(), 
        (mean_opt - 2*std_opt).squeeze(), 
        (mean_opt + 2*std_opt).squeeze(), 
        alpha=0.3, 
        label='95% confidence',
        zorder=1
    )
    ax2.plot(X_test, myFunc(X_test), 'r--', label='True function', linewidth=1.5, zorder=2)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('With Hyperparameter Optimization', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.05, 0.95, f'NLML = {nlml_opt:.2f}', 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.suptitle('Gaussian Process Regression Comparison', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    # ____Save the figure____
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saveDir = "figs"
    figPath = f'{saveDir}/gp_comparison_{timestamp}.png'
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    plt.savefig(figPath, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved as '{figPath}'")
