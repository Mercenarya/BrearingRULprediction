"""
Tests for vibration encoder.
"""
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.vibration_encoder import VibrationEncoder
from models.config import ModelConfig


def test_vibration_encoder_output_shape(model_config, synthetic_batch):
    """Test output shape of vibration encoder."""
    encoder = VibrationEncoder(model_config)
    vib_input = synthetic_batch["vibration"]
    features = encoder(vib_input)
    assert isinstance(features, dict)
    assert set(features.keys()) == {"early", "mid", "late"}
    for key, tensor in features.items():
        assert tensor.shape[:2] == vib_input.shape[:2]  # [B, T, ...]
        assert tensor.shape[-1] == model_config.vibration_feature_dim