"""
Evaluation metrics for Bearing RUL prediction.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error.

    Args:
        y_true (np.ndarray): True RUL values
        y_pred (np.ndarray): Predicted RUL values

    Returns:
        float: MAE value
    """
    return mean_absolute_error(y_true, y_pred)

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Square Error.

    Args:
        y_true (np.ndarray): True RUL values
        y_pred (np.ndarray): Predicted RUL values

    Returns:
        float: RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_r2(y_true, y_pred):
    """
    Calculate R-squared (coefficient of determination).

    Args:
        y_true (np.ndarray): True RUL values
        y_pred (np.ndarray): Predicted RUL values

    Returns:
        float: R² value
    """
    return r2_score(y_true, y_pred)

def calculate_phm_score(y_true, y_pred):
    """
    Calculate PHM Score as defined in the PHM 2012 challenge.
    Lower is better, with perfect score being 0.

    Args:
        y_true (np.ndarray): True RUL values
        y_pred (np.ndarray): Predicted RUL values

    Returns:
        float: PHM Score
    """
    # Convert to numpy arrays if needed
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate errors
    errors = y_pred - y_true

    # Create weight arrays based on error sign
    # For under-prediction (errors < 0): use exponential weighting
    # For over-prediction (errors >= 0): use exponential weighting
    weights = np.where(
        errors < 0,
        np.exp(-0.1 * errors) - 1,  # Under-prediction
        np.exp(0.1 * errors) - 1    # Over-prediction
    )

    # Calculate PHM score
    phm_score = np.sum(weights) / len(y_true)

    return phm_score

def calculate_all_metrics(y_true, y_pred):
    """
    Calculate all evaluation metrics.

    Args:
        y_true (np.ndarray): True RUL values
        y_pred (np.ndarray): Predicted RUL values

    Returns:
        dict: Dictionary containing all metrics
    """
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    phm_score = calculate_phm_score(y_true, y_pred)

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'phm_score': phm_score
    }

if __name__ == "__main__":
    # Example usage
    print("Metrics module ready")