"""
Tests for scale fusion.
"""
import torch
from models.scale_fusion import ScaleFusion
from models.config import ModelConfig


def test_scale_fusion_concat_project(model_config):
    """Test scale fusion with concat_project mode."""
    model_config.fusion_type = "concat_project"
    fusion = ScaleFusion(model_config)
    B, T, D = 4, 32, model_config.d_model
    features = {
        "early": torch.randn(B, T, D),
        "mid": torch.randn(B, T, D),
        "late": torch.randn(B, T, D),
    }
    output = fusion(features)
    assert output.shape == (B, T, D)


def test_scale_fusion_weighted_sum_global(model_config):
    """Test scale fusion with weighted_sum global mode."""
    model_config.fusion_type = "weighted_sum"
    model_config.scale_weight_type = "global"
    fusion = ScaleFusion(model_config)
    B, T, D = 4, 32, model_config.d_model
    features = {
        "early": torch.randn(B, T, D),
        "mid": torch.randn(B, T, D),
        "late": torch.randn(B, T, D),
    }
    output = fusion(features)
    assert output.shape == (B, T, D)


def test_scale_fusion_weighted_sum_dynamic(model_config):
    """Test scale fusion with weighted_sum dynamic mode."""
    model_config.fusion_type = "weighted_sum"
    model_config.scale_weight_type = "dynamic"
    fusion = ScaleFusion(model_config)
    B, T, D = 4, 32, model_config.d_model
    features = {
        "early": torch.randn(B, T, D),
        "mid": torch.randn(B, T, D),
        "late": torch.randn(B, T, D),
    }
    output = fusion(features)
    assert output.shape == (B, T, D)