"""
Tests for temperature encoder.
"""
import torch
from models.temperature_encoder import TemperatureEncoder
from models.config import ModelConfig


def test_temperature_encoder_feature_vector(model_config, synthetic_batch):
    """Test temperature encoder with feature vector representation."""
    model_config.temp_representation = "feature_vector"
    encoder = TemperatureEncoder(model_config)
    temp_input = synthetic_batch["temperature"]
    temp_mask = synthetic_batch["temperature_mask"]
    features = encoder(temp_input, temp_mask)
    assert isinstance(features, dict)
    assert set(features.keys()) == {"early", "mid", "late"}
    for key, tensor in features.items():
        assert tensor.shape[:2] == temp_input.shape[:2]  # [B, T, ...]
        assert tensor.shape[-1] == model_config.temperature_feature_dim


def test_temperature_encoder_history(model_config, synthetic_batch):
    """Test temperature encoder with history representation."""
    model_config.temp_representation = "history"
    model_config.temp_history_length = 64
    encoder = TemperatureEncoder(model_config)
    # Create history-shaped input
    B, T, _ = synthetic_batch["temperature"].shape
    temp_input = torch.randn(B, T, 1, model_config.temp_history_length)
    temp_mask = torch.ones(B, T, 1, model_config.temp_history_length, dtype=torch.bool)
    features = encoder(temp_input, temp_mask)
    assert isinstance(features, dict)
    assert set(features.keys()) == {"early", "mid", "late"}
    for key, tensor in features.items():
        assert tensor.shape[:2] == (B, T)
        assert tensor.shape[-1] == model_config.temperature_feature_dim