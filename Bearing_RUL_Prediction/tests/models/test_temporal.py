"""
Tests for temporal encoder.
"""
import torch
from models.temporal import TemporalEncoder, DualPathTemporalEncoder
from models.config import ModelConfig


def test_temporalencoder_gru(model_config):
    """Test TemporalEncoder with GRU."""
    model_config.temporal_type = "gru"
    encoder = TemporalEncoder(model_config)
    B, T, D = 4, 32, model_config.d_model
    x = torch.randn(B, T, D)
    output, hidden, cell = encoder(x)
    assert output.shape == (B, T, model_config.temporal_hidden_size * (1 if not model_config.bidirectional else 2))
    assert hidden is not None
    assert cell is None


def test_temporalencoder_lstm(model_config):
    """Test TemporalEncoder with LSTM."""
    model_config.temporal_type = "lstm"
    encoder = TemporalEncoder(model_config)
    B, T, D = 4, 32, model_config.d_model
    x = torch.randn(B, T, D)
    output, hidden, cell = encoder(x)
    assert output.shape == (B, T, model_config.temporal_hidden_size * (1 if not model_config.bidirectional else 2))
    assert hidden is not None
    assert cell is not None


def test_dualpathtemporalencoder(model_config):
    """Test DualPathTemporalEncoder."""
    encoder = DualPathTemporalEncoder(model_config)
    B, T, D = 4, 32, model_config.d_model
    x = torch.randn(B, T, D)
    output, hidden, cell = encoder(x)
    assert output.shape == (B, T, model_config.d_model)
    # hidden and cell are None in current implementation
    assert hidden is None
    assert cell is None