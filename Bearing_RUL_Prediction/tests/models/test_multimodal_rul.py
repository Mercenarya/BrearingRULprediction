"""
Tests for multimodal RUL model.
"""
import torch
from models.multimodal_rul import MultimodalRUL
from models.config import ModelConfig


def test_multimodal_rul(model_config, synthetic_batch):
    """Test MultimodalRUL forward pass."""
    model = MultimodalRUL(model_config)
    output = model(synthetic_batch)
    assert "rul_pred" in output
    assert output["rul_pred"].shape == (synthetic_batch["vibration"].shape[0], 1)
    if model_config.return_aux:
        assert output["aux"] is not None
    else:
        assert output["aux"] is None