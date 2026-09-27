"""
PyTorch Dataset and DataLoader for bearing RUL prediction.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class BearingDataset(Dataset):
    """
    Dataset class for bearing vibration and temperature data with RUL labels.
    """

    def __init__(self, data_path, transform=None):
        """
        Initialize the bearing dataset.

        Args:
            data_path (str): Path to the .npz file containing data and labels
            transform (callable, optional): Optional transform to be applied on samples
        """
        loaded_data = np.load(data_path)
        self.vibration = loaded_data['vibration']  # Shape: (n_samples, window_size, n_vib_channels)
        self.temperature = loaded_data['temperature']  # Shape: (n_samples, window_size, n_temp_channels)
        self.rul = loaded_data['rul']  # Shape: (n_samples,)
        self.transform = transform

        # Ensure data is float32 for PyTorch
        self.vibration = self.vibration.astype(np.float32)
        self.temperature = self.temperature.astype(np.float32)
        self.rul = self.rul.astype(np.float32)

    def __len__(self):
        return len(self.rul)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index

        Returns:
            dict: Dictionary containing vibration, temperature, and RUL label
        """
        vibration_sample = self.vibration[idx]
        temperature_sample = self.temperature[idx]
        rul_label = self.rul[idx]

        sample = {
            'vibration': vibration_sample,
            'temperature': temperature_sample,
            'rul': rul_label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

def create_data_loaders(train_path, val_path, test_path, batch_size, num_workers=0):
    """
    Create train, validation, and test DataLoaders.

    Args:
        train_path (str): Path to training data .npz file
        val_path (str): Path to validation data .npz file
        test_path (str): Path to test data .npz file
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of subprocesses for data loading

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = BearingDataset(train_path)
    val_dataset = BearingDataset(val_path)
    test_dataset = BearingDataset(test_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Example usage
    print("Dataset module ready")