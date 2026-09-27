"""
Create sliding windows from preprocessed bearing data.
"""
import numpy as np

def create_sliding_windows(data, window_size, step_size):
    """
    Create sliding windows from 1D or 2D signal data.

    Args:
        data (np.ndarray): Input signal data of shape (n_samples,) or (n_samples, n_channels)
        window_size (int): Size of each window
        step_size (int): Step between consecutive windows

    Returns:
        np.ndarray: Windowed data of shape (n_windows, window_size) or (n_windows, window_size, n_channels)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_channels = data.shape
    n_windows = (n_samples - window_size) // step_size + 1

    windows = np.zeros((n_windows, window_size, n_channels))

    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        windows[i] = data[start_idx:end_idx]

    # Return 3D array if multi-channel, otherwise squeeze to 2D
    if n_channels == 1:
        return windows.squeeze(-1)
    return windows

def create_multichannel_windows(vibration_data, temperature_data, window_size, step_size):
    """
    Create windows combining vibration and temperature data.

    Args:
        vibration_data (np.ndarray): Vibration signal of shape (n_samples,) or (n_samples, n_vib_channels)
        temperature_data (np.ndarray): Temperature signal of shape (n_samples,) or (n_samples, n_temp_channels)
        window_size (int): Size of each window
        step_size (int): Step between consecutive windows

    Returns:
        np.ndarray: Combined windowed data of shape (n_windows, window_size, n_total_channels)
    """
    # Ensure both are at least 2D
    if vibration_data.ndim == 1:
        vibration_data = vibration_data.reshape(-1, 1)
    if temperature_data.ndim == 1:
        temperature_data = temperature_data.reshape(-1, 1)

    # Create windows for each modality
    vib_windows = create_sliding_windows(vibration_data, window_size, step_size)
    temp_windows = create_sliding_windows(temperature_data, window_size, step_size)

    # Concatenate along channel dimension
    combined_windows = np.concatenate([vib_windows, temp_windows], axis=-1)
    return combined_windows

if __name__ == "__main__":
    # Example usage
    print("Sliding window module ready")