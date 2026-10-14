"""
Script to validate bearing dataset .npz files.
"""
import os
import numpy as np
import hashlib
from typing import Dict, List, Tuple


def validate_npz_file(file_path: str) -> Tuple[bool, List[str]]:
    """
    Validate a single .npz file.

    Args:
        file_path: Path to the .npz file

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []

    # Check if file exists
    if not os.path.exists(file_path):
        errors.append(f"File does not exist: {file_path}")
        return False, errors

    # Try to load the file
    try:
        data = np.load(file_path)
    except Exception as e:
        errors.append(f"Failed to load .npz file: {e}")
        return False, errors

    # Check required keys
    required_keys = {'vibration', 'temperature', 'rul'}
    present_keys = set(data.keys())
    missing_keys = required_keys - present_keys
    if missing_keys:
        errors.append(f"Missing required keys: {missing_keys}")

    # If we have the keys, check their properties
    if not missing_keys:
        # Check vibration
        vib = data['vibration']
        if vib.dtype != np.float32:
            errors.append(f"Vibration dtype should be float32, got {vib.dtype}")
        if vib.ndim != 3:
            errors.append(f"Vibration should be 3D array (samples, window, channels), got shape {vib.shape}")
        else:
            n_samples, window_size, n_channels = vib.shape
            if window_size != 1024:
                errors.append(f"Expected window size 1024, got {window_size}")
            if n_channels != 2:
                errors.append(f"Expected 2 vibration channels, got {n_channels}")

        # Check temperature
        temp = data['temperature']
        if temp.dtype != np.float32:
            errors.append(f"Temperature dtype should be float32, got {temp.dtype}")
        if temp.ndim != 3:
            errors.append(f"Temperature should be 3D array (samples, window, channels), got shape {temp.shape}")
        else:
            n_samples_temp, window_size_temp, n_channels_temp = temp.shape
            if window_size_temp != 1024:
                errors.append(f"Expected window size 1024 for temperature, got {window_size_temp}")
            if n_channels_temp != 1:
                errors.append(f"Expected 1 temperature channel, got {n_channels_temp}")

        # Check RUL
        rul = data['rul']
        if rul.dtype != np.float32:
            errors.append(f"RUL dtype should be float32, got {rul.dtype}")
        if rul.ndim != 1:
            errors.append(f"RUL should be 1D array (samples,), got shape {rul.shape}")

        # Check consistency of number of samples
        if 'vibration' in data and 'temperature' in data and 'rul' in data:
            n_samples_vib = data['vibration'].shape[0]
            n_samples_temp = data['temperature'].shape[0]
            n_samples_rul = data['rul'].shape[0]
            if not (n_samples_vib == n_samples_temp == n_samples_rul):
                errors.append(f"Inconsistent number of samples: vibration={n_samples_vib}, temperature={n_samples_temp}, rul={n_samples_rul}")

        # Check for NaNs or infs
        for key in required_keys:
            arr = data[key]
            if np.isnan(arr).any():
                errors.append(f"Array '{key}' contains NaN values")
            if np.isinf(arr).any():
                errors.append(f"Array '{key}' contains infinite values")

    # Close the file
    data.close()

    return len(errors) == 0, errors


def compute_file_hash(file_path: str) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        SHA256 hash as hex string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def validate_dataset_directory(data_dir: str) -> Dict[str, Dict]:
    """
    Validate all .npz files in a directory.

    Args:
        data_dir: Directory containing .npz files

    Returns:
        Dictionary with validation results for each file
    """
    results = {}
    if not os.path.isdir(data_dir):
        print(f"Directory does not exist: {data_dir}")
        return results

    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    if not npz_files:
        print(f"No .npz files found in {data_dir}")
        return results

    for file_name in npz_files:
        file_path = os.path.join(data_dir, file_name)
        is_valid, errors = validate_npz_file(file_path)
        file_hash = compute_file_hash(file_path)
        results[file_name] = {
            'valid': is_valid,
            'errors': errors,
            'sha256': file_hash,
            'size_bytes': os.path.getsize(file_path)
        }
        if is_valid:
            print(f"[VALID] {file_name}: Valid")
        else:
            print(f"[INVALID] {file_name}: Invalid")
            for error in errors:
                print(f"  - {error}")

    return results


if __name__ == "__main__":
    # Validate the data/processed directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    data_dir = os.path.abspath(data_dir)
    print(f"Validating datasets in: {data_dir}")
    print("="*60)
    results = validate_dataset_directory(data_dir)

    # Summary
    valid_count = sum(1 for v in results.values() if v['valid'])
    total_count = len(results)
    print("="*60)
    print(f"Summary: {valid_count}/{total_count} files are valid.")

    # Optionally, save results to a JSON file
    import json
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'validation_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {output_path}")