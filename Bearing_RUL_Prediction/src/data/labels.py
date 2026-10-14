"""
Calculate RUL labels and normalize according to max_RUL.
"""
import numpy as np

# hàm tính toán phạt (thời gian hạot động, thời gian hiện tại)
def calculate_rul_from_failure(lifetimes, current_time):
    """
    Calculate Remaining Useful Life based on failure times.

    Args:
        lifetimes (np.ndarray): Array of failure times for each component
        current_time (np.ndarray): Array of current times

    Returns:
        np.ndarray: RUL values (lifetimes - current_time)
    """
    return np.maximum(0, lifetimes - current_time)


# chuẩn hoá giá trị được nhập vào
def normalize_rul(rul_values, max_rul=None):
    """
    Normalize RUL values by dividing by max_RUL.

    Args:
        rul_values (np.ndarray): Raw RUL values
        max_rul (float, optional): Maximum RUL for normalization.
                                  If None, uses max of rul_values.

    Returns:
        np.ndarray: Normalized RUL values in [0, 1]
    """
    if max_rul is None:
        max_rul = np.max(rul_values)

    if max_rul == 0:
        return np.zeros_like(rul_values)

    return rul_values / max_rul

def denormalize_rul(normalized_rul, max_rul):
    """
    Convert normalized RUL back to original scale.

    Args:
        normalized_rul (np.ndarray): Normalized RUL values [0, 1]
        max_rul (float): Maximum RUL used for normalization

    Returns:
        np.ndarray: RUL values in original scale
    """
    return normalized_rul * max_rul

def process_bearing_lifetimes(raw_data, failure_threshold=0.1):
    """
    Process raw bearing data to extract lifetimes and compute RUL.

    Args:
        raw_data (dict): Dictionary containing bearing measurements over time
        failure_threshold (float): Threshold for determining failure

    Returns:
        tuple: (lifetimes, rul_values) where lifetimes is time to failure
               and rul_values is remaining useful life at each time point
    """
    # This is a simplified implementation - actual implementation would
    # depend on the specific dataset format
    # For demonstration purposes:
    n_samples = len(raw_data.get('vibration', []))
    lifetimes = np.full(n_samples, 100.0)  # Placeholder: 100 time units to failure
    current_times = np.arange(n_samples)   # Current time points
    rul_values = calculate_rul_from_failure(lifetimes, current_times)
    return lifetimes, rul_values

if __name__ == "__main__":
    # Example usage
    print("Labels module ready")