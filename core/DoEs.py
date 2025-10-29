import numpy as np

def sobol_g(a: np.ndarray):
    """
    Sobol G-function for sensitivity analysis.
    f(x) = prod((abs(4.0 * x_i - 2.0) + a_i) / (1.0 + a_i))
    
    Args:
        a: Array of shape (d,)
    
    Returns:
        Function that takes x (array of length d) and returns {"y": value}
    """
    a = np.asarray(a, dtype=float)
    d = a.size
    def f(x):
        x = np.asarray(x, dtype=float)
        val = 1.0
        for i in range(d):
            val *= (abs(4.0 * x[i] - 2.0) + a[i]) / (1.0 + a[i])
        return {"y": float(val)}
    return f


def morris_g(k: int):
    """
    Morris function for sensitivity analysis.
    f(x) = sum(alpha * x_i) + sum(beta * x_i * x_j) for i < j
    
    Args:
        k: Dimensionality (number of input variables)
    
    Returns:
        Function that takes x (array of length k) and returns {"y": value}
    """
    alpha = np.sqrt(12) - 6*np.sqrt(0.1*(k-1))
    beta = 12*np.sqrt(0.1) / np.sqrt(k-1)
    
    def f(x):
        x = np.asarray(x, dtype=float)
        
        # Linear term: sum of alpha * x_i
        linear_term = np.sum(alpha * x)
        
        # Interaction term: sum of beta * x_i * x_j for i < j
        interaction_term = 0.0
        for i in range(k):
            for j in range(i+1, k):
                interaction_term += beta * x[i] * x[j]
        
        val = linear_term + interaction_term
        return {"y": float(val)}
        
    return f


