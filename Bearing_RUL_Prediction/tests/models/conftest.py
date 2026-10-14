"""
Pytest configuration for model tests.
"""
import pytest
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.config import ModelConfig


@pytest.fixture
def model_config():
    """Default model configuration for testing."""
    return ModelConfig()


@pytest.fixture
def synthetic_batch():
    """Synthetic batch for testing."""
    batch_size = 4
    sequence_length = 32
    vibration_length = 1024
    temperature_features = 7
    return {
        "vibration": torch.randn(batch_size, sequence_length, 2, vibration_length),
        "temperature": torch.randn(batch_size, sequence_length, temperature_features),
        "temperature_mask": torch.ones(batch_size, sequence_length, temperature_features, dtype=torch.bool),
        "rul": torch.rand(batch_size, 1) * 1000,
    }