"""
Tests for cross-modal attention.
"""
import torch
from models.cross_attention import CrossModalAttention
from models.config import ModelConfig


def test_cross_modal_attention(model_config, synthetic_batch):
    """Test cross-modal attention forward pass."""
    # Create dummy features matching encoder output shape
    B, T, _, _ = synthetic_batch["vibration"].shape
    D_v = model_config.vibration_feature_dim
    D_t = model_config.temperature_feature_dim
    D_model = model_config.d_model

    vib_features = {
        "early": torch.randn(B, T, D_v),
        "mid": torch.randn(B, T, D_v),
        "late": torch.randn(B, T, D_v),
    }
    temp_features = {
        "early": torch.randn(B, T, D_t),
        "mid": torch.randn(B, T, D_t),
        "late": torch.randn(B, T, D_t),
    }
    temp_mask = synthetic_batch["temperature_mask"]

    attn = CrossModalAttention(model_config)
    vib_attended, temp_attended = attn(vib_features, temp_features, temperature_mask=temp_mask)

    assert isinstance(vib_attended, dict)
    assert isinstance(temp_attended, dict)
    assert set(vib_attended.keys()) == {"early", "mid", "late"}
    assert set(temp_attended.keys()) == {"early", "mid", "late"}
    for key in ["early", "mid", "late"]:
        assert vib_attended[key].shape == (B, T, D_model)
        assert temp_attended[key].shape == (B, T, D_model)