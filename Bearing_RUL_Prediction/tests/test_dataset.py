"""
Unit tests for the BearingDataset class and data loading functions.
"""
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader
import tempfile
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import BearingDataset, create_data_loaders

class TestBearingDataset(unittest.TestCase):
    """Test cases for BearingDataset class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary .npz file with test data
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
        self.temp_file.close()

        # Create test data
        n_samples = 100
        vibration_data = np.random.randn(n_samples, 64, 3).astype(np.float32)  # (samples, window, channels)
        temperature_data = np.random.randn(n_samples, 64, 1).astype(np.float32)
        rul_data = np.random.rand(n_samples).astype(np.float32) * 100  # RUL values 0-100

        # Save to temporary file
        np.savez(self.temp_file.name,
                 vibration=vibration_data,
                 temperature=temperature_data,
                 rul=rul_data)

        self.dataset = BearingDataset(self.temp_file.name)

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        os.unlink(self.temp_file.name)

    def test_dataset_length(self):
        """Test that dataset length matches number of samples."""
        self.assertEqual(len(self.dataset), 100)

    def test_dataset_getitem(self):
        """Test that getitem returns correct data types and shapes."""
        sample = self.dataset[0]

        # Check that sample is a dictionary with expected keys
        self.assertIsInstance(sample, dict)
        self.assertIn('vibration', sample)
        self.assertIn('temperature', sample)
        self.assertIn('rul', sample)

        # Check data types
        self.assertIsInstance(sample['vibration'], np.ndarray)
        self.assertIsInstance(sample['temperature'], np.ndarray)
        self.assertIsInstance(sample['rul'], np.ndarray)

        # Check shapes
        self.assertEqual(sample['vibration'].shape, (64, 3))  # (window_size, vib_channels)
        self.assertEqual(sample['temperature'].shape, (64, 1))  # (window_size, temp_channels)
        self.assertEqual(sample['rul'].shape, ())  # scalar

        # Check data types are float32
        self.assertEqual(sample['vibration'].dtype, np.float32)
        self.assertEqual(sample['temperature'].dtype, np.float32)
        self.assertEqual(sample['rul'].dtype, np.float32)

    def test_dataset_indexing_bounds(self):
        """Test that indexing respects bounds."""
        # Valid indices
        _ = self.dataset[0]
        _ = self.dataset[99]

        # Invalid indices should raise IndexError
        with self.assertRaises(IndexError):
            _ = self.dataset[-1]

        with self.assertRaises(IndexError):
            _ = self.dataset[100]

class TestDataLoaders(unittest.TestCase):
    """Test cases for data loader creation functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory and files
        self.temp_dir = tempfile.mkdtemp()

        # Create train, val, test files
        self.train_file = os.path.join(self.temp_dir, 'train.npz')
        self.val_file = os.path.join(self.temp_dir, 'val.npz')
        self.test_file = os.path.join(self.temp_dir, 'test.npz')

        # Create test data for each file
        n_samples = 50
        for file_path in [self.train_file, self.val_file, self.test_file]:
            vibration_data = np.random.randn(n_samples, 32, 2).astype(np.float32)
            temperature_data = np.random.randn(n_samples, 32, 1).astype(np.float32)
            rul_data = np.random.rand(n_samples).astype(np.float32) * 50

            np.savez(file_path,
                     vibration=vibration_data,
                     temperature=temperature_data,
                     rul=rul_data)

    def tearDown(self):
        """Tear down test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_create_data_loaders(self):
        """Test that data loaders are created correctly."""
        train_loader, val_loader, test_loader = create_data_loaders(
            self.train_file, self.val_file, self.test_file,
            batch_size=10,
            num_workers=0
        )

        # Check that loaders are DataLoader instances
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)

        # Check batch size
        self.assertEqual(train_loader.batch_size, 10)
        self.assertEqual(val_loader.batch_size, 10)
        self.assertEqual(test_loader.batch_size, 10)

        # Check number of batches
        self.assertEqual(len(train_loader), 5)  # 50 samples / 10 batch_size
        self.assertEqual(len(val_loader), 5)
        self.assertEqual(len(test_loader), 5)

    def test_data_loader_iteration(self):
        """Test that we can iterate through data loaders and get correct batch shapes."""
        train_loader, _, _ = create_data_loaders(
            self.train_file, self.val_file, self.test_file,
            batch_size=8,
            num_workers=0
        )

        # Get first batch
        for batch in train_loader:
            vibration_batch = batch['vibration']
            temperature_batch = batch['temperature']
            rul_batch = batch['rul']

            # Check batch dimensions
            self.assertEqual(vibration_batch.shape, (8, 32, 2))  # (batch, window, vib_channels)
            self.assertEqual(temperature_batch.shape, (8, 32, 1))  # (batch, window, temp_channels)
            self.assertEqual(rul_batch.shape, (8, 1))  # (batch, 1)

            # Check data types
            self.assertEqual(vibration_batch.dtype, torch.float32)
            self.assertEqual(temperature_batch.dtype, torch.float32)
            self.assertEqual(rul_batch.dtype, torch.float32)

            break  # Only test first batch

if __name__ == '__main__':
    unittest.main()