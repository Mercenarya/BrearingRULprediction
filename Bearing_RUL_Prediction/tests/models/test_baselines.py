"""
Tests for baseline models.
"""
import torch
from models.baselines import (
    VibrationCNNRegressor,
    TemperatureRegressor,
    LateConcatFusionRegressor,
    CNNTemporalRegressor
)
from models.config import ModelConfig


def test_vibration_cnn_regressor(model_config, synthetic_batch):
    """Test VibrationCNNRegressor forward pass."""
    model = VibrationCNNRegressor(model_config)
    output = model(synthetic_batch)
    assert "rul_pred" in output
    assert output["rul_pred"].shape == (synthetic_batch["vibration"].shape[0], 1)
    if model_config.return_aux:
        assert output["aux"] is not None
    else:
        assert output["aux"] is None


def test_temperature_regressor(model_config, synthetic_batch):
    """Test TemperatureRegressor forward pass."""
    model = TemperatureRegressor(model_config)
    output = model(synthetic_batch)
    assert "rul_pred" in output
    assert output["rul_pred"].shape == (synthetic_batch["vibration"].shape[0], 1)
    if model_config.return_aux:
        assert output["aux"] is not None
    else:
        assert output["aux"] is None


def test_late_concat_fusion_regressor(model_config, synthetic_batch):
    """Test LateConcatFusionRegressor forward pass."""
    model = LateConcatFusionRegressor(model_config)
    output = model(synthetic_batch)
    assert "rul_pred" in output
    assert output["rul_pred"].shape == (synthetic_batch["vibration"].shape[0], 1)
    if model_config.return_aux:
        assert output["aux"] is not None
    else:
        assert output["aux"] is None


def test_cnn_temporal_regressor(model_config, synthetic_batch):
    """Test CNNTemporalRegressor forward pass."""
    model = CNNTemporalRegressor(model_config)
    output = model(synthetic_batch)
    assert "rul_pred" in output
    assert output["rul_pred"].shape == (synthetic_batch["vibration"].shape[0], 1)
    if model_config.return_aux:
        assert output["aux"] is not None
    else:
        assert output["aux"] is None