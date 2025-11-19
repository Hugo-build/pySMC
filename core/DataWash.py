import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler


# =============================================================
#                         RMV data 
# =============================================================

def remove_zeros(X: np.ndarray, y: np.ndarray, threshold:tuple[float, float] = (1e-4, 1e-4)) -> Tuple[np.ndarray, np.ndarray]:
    """Remove rows with zeros from X and y.
    """
    mask4X = np.all(np.abs(X) > threshold, axis=1)
    if y.ndim == 1:
        mask4y = np.all(np.abs(y) > threshold)
    else:
        mask4y = np.all(np.abs(y) > threshold, axis=1)
    mask4y = np.all(np.abs(y) > threshold)

    return X[mask4X], y[mask4y]

def remove_nan(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove rows with NaNs from X and y.
    """
    mask4X = np.all(np.isfinite(X), axis=1)
    if y.ndim == 1:
        mask4y = np.all(np.isfinite(y))
    else:
        mask4y = np.all(np.isfinite(y), axis=1)
    mask4y = np.all(np.isfinite(y))
    return X[mask4X], y[mask4y]


# =============================================================
#                         Split data 
# =============================================================

def train_test_split(X: np.ndarray, y: np.ndarray, 
                       test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training and testing sets.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,) or (n_samples, 1)
    test_size : float
    random_state : int
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    n_test = int(test_size * n_samples)
    n_train = n_samples - n_test
    idx = rng.permutation(n_samples)
    X_train = X[idx[:n_train]]
    y_train = y[idx[:n_train]]
    X_test = X[idx[n_train:]]
    y_test = y[idx[n_train:]]
    return X_train, X_test, y_train, y_test


def train_test_validate_split(X: np.ndarray, y: np.ndarray, 
                              test_size: float = 0.2, validate_size: float = 0.2, 
                              random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training, testing, and validation sets.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,) or (n_samples, 1)
    test_size : float
    validate_size : float
    random_state : int
    """
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    n_test = int(test_size * n_samples)
    n_validate = int(validate_size * n_samples)
    n_train = n_samples - n_test - n_validate
    idx = rng.permutation(n_samples)
    X_train = X[idx[:n_train]]
    y_train = y[idx[:n_train]]
    X_test = X[idx[n_train:n_train+n_test]]
    y_test = y[idx[n_train:n_train+n_test]]
    X_validate = X[idx[n_train+n_test:]]
    y_validate = y[idx[n_train+n_test:]]
    return X_train, X_test, X_validate, y_train, y_test, y_validate


# =============================================================
#                         Scale data 
# =============================================================

def scale_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """Standardize features and targets (zero mean, unit variance).

    Returns scaled arrays and the fitted scalers for X and y.
    """
    # Fit separate scalers for X and y
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)

    y = np.asarray(y)
    y_scaler = StandardScaler()
    if y.ndim == 1:
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
    else:
        y_scaled = y_scaler.fit_transform(y)
    return X_scaled, y_scaled, x_scaler, y_scaler

def unscale_data(X_scaled: np.ndarray, y_scaled: np.ndarray, 
                 x_scaler: StandardScaler, y_scaler: StandardScaler) -> Tuple[np.ndarray, np.ndarray]:
    """Invert the standardization for features and targets.
    """
    X = x_scaler.inverse_transform(X_scaled)
    if y_scaled.ndim == 1:
        y = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    else:
        y = y_scaler.inverse_transform(y_scaled)
    return X, y