"""
Bandpass filter, downsample, and normalize bearing vibration data.
"""
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply bandpass filter to signal.

    Args:
        data (np.ndarray): Input signal
        lowcut (float): Low cutoff frequency
        highcut (float): High cutoff frequency
        fs (float): Sampling frequency
        order (int): Filter order

    Returns:
        np.ndarray: Filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def downsample_signal(data, original_fs, target_fs):
    """
    Downsample signal from original_fs to target_fs.

    Args:
        data (np.ndarray): Input signal
        original_fs (float): Original sampling frequency
        target_fs (float): Target sampling frequency

    Returns:
        np.ndarray: Downsampled signal
    """
    if target_fs >= original_fs:
        return data

    # Calculate decimation factor
    dec_factor = int(original_fs / target_fs)
    return signal.decimate(data, dec_factor, ftype='iir')

def normalize_data(data, method="standard"):
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Reshape for sklearn scalers
    original_shape = data.shape
    data_reshaped = data.reshape(-1, original_shape[-1])
    data_normalized = scaler.fit_transform(data_reshaped)
    return data_normalized.reshape(original_shape)

def preprocess_signal(raw_signal, config):
    """
    Apply full preprocessing pipeline to raw signal.

    Args:
        raw_signal (np.ndarray): Raw vibration/temperature signal
        config (dict): Configuration parameters

    Returns:
        np.ndarray: Preprocessed signal
    """
    # Apply bandpass filter
    filtered = bandpass_filter(
        raw_signal,
        lowcut=config.get('lowcut', 100),
        highcut=config.get('highcut', 5000),
        fs=config.get('sampling_rate', 12000)
    )

    # Downsample if needed
    if config.get('target_fs'):
        filtered = downsample_signal(
            filtered,
            original_fs=config.get('sampling_rate', 12000),
            target_fs=config['target_fs']
        )

    # Normalize
    normalized = normalize_data(
        filtered,
        method=config.get('normalization', 'standard')
    )

    return normalized

if __name__ == "__main__":
    # Example usage
    print("Preprocessing module ready")