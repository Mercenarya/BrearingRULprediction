"""
Unit tests for verifying tensor shapes match MODEL_WORKFLOW_DETAILED.md specifications.
"""
import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.vibration_encoder import VibrationEncoder
from models.temperature_encoder import TemperatureEncoder
from models.cross_attention import CrossModalAttention
from models.scale_fusion import ScaleFusion
from models.temporal import DualPathTemporalEncoder
from models.regression_head import RegressionHead
from models.rul_model import RULModel

class TestModelShapes(unittest.TestCase):
    """Test cases for verifying model tensor shapes."""

    def setUp(self):
        """Set up test fixtures with default parameters from MODEL_WORKFLOW_DETAILED_DETAILED.md."""
        # Default parameters based on documentation
        self.batch_size = 32
        self.seq_len = 1024  # window size
        self.vib_in_channels = 3  # 3-axis vibration
        self.temp_in_channels = 1  # temperature
        self.cnn_out_channels = 64
        self.lstm_hidden_size = 128
        self.fusion_type = "concat"
        self.num_lstm_layers = 2

        # Create test input tensors
        self.vib_input = torch.randn(self.batch_size, self.seq_len, self.vib_in_channels)
        self.temp_input = torch.randn(self.batch_size, self.seq_len, self.temp_in_channels)

    def test_cnn_encoder_shapes(self):
        """Test CNN encoder output shapes."""
        # Test vibration encoder
        vib_encoder = VibrationEncoder(self.vib_in_channels, self.cnn_out_channels)

        # CNN expects (batch, channels, seq_len) input
        vib_transposed = self.vib_input.transpose(1, 2)  # (batch, vib_in_channels, seq_len)
        vib_features = vib_encoder(vib_transposed)  # (batch, 3*out_channels, seq_len)

        # Expected shape after CNN: [batch, 3*cnn_out_channels, seq_len]
        expected_shape = (self.batch_size, 3 * self.cnn_out_channels, self.seq_len)
        self.assertEqual(vib_features.shape, expected_shape)

        # Test temperature encoder
        temp_encoder = TemperatureEncoder(self.temp_in_channels, self.cnn_out_channels)
        temp_transposed = self.temp_input.transpose(1, 2)
        temp_features = temp_encoder(temp_transposed)

        expected_shape = (self.batch_size, 3 * self.cnn_out_channels, self.seq_len)
        self.assertEqual(temp_features.shape, expected_shape)

    def test_cross_modal_attention_shapes(self):
        """Test cross-modal attention output shapes."""
        # Features after CNN encoding (before transpose back to LSTM format)
        feature_dim = 3 * self.cnn_out_channels  # 192

        vib_features = torch.randn(self.batch_size, self.seq_len, feature_dim)
        temp_features = torch.randn(self.batch_size, self.seq_len, feature_dim)

        cross_attn = CrossModalAttention(feature_dim=feature_dim, num_heads=4)
        vib_attended, temp_attended = cross_attn(vib_features, temp_features)

        # Output shapes should match input shapes
        self.assertEqual(vib_attended.shape, vib_features.shape)
        self.assertEqual(temp_attended.shape, temp_features.shape)

        expected_shape = (self.batch_size, self.seq_len, feature_dim)
        self.assertEqual(vib_attended.shape, expected_shape)
        self.assertEqual(temp_attended.shape, expected_shape)

    def test_fusion_strategy_shapes(self):
        """Test fusion strategy output shapes."""
        feature_dim = 3 * self.cnn_out_channels  # 192

        vib_features = torch.randn(self.batch_size, self.seq_len, feature_dim)
        temp_features = torch.randn(self.batch_size, self.seq_len, feature_dim)

        # Test concat fusion (as implemented in documentation)
        fusion = FusionStrategy(
            vib_feature_dim=feature_dim,
            temp_feature_dim=feature_dim,
            fusion_type="concat"
        )

        fused_features = fusion(vib_features, temp_features)

        # For concat fusion, output should be [batch, seq_len, 2*feature_dim]
        expected_shape = (self.batch_size, self.seq_len, 2 * feature_dim)
        self.assertEqual(fused_features.shape, expected_shape)

        # Test other fusion types for completeness
        for fusion_type in ["early", "mid", "late"]:
            fusion = FusionStrategy(
                vib_feature_dim=feature_dim,
                temp_feature_dim=feature_dim,
                fusion_type=fusion_type
            )

            # For early/mid/late, output should match vib_feature_dim (after processing)
            fused_features = fusion(vib_features, temp_features)
            expected_shape = (self.batch_size, self.seq_len, feature_dim)
            self.assertEqual(fused_features.shape, expected_shape)

    def test_dual_path_lstm_shapes(self):
        """Test Dual-Path LSTM output shapes."""
        # Input size depends on fusion type
        if self.fusion_type == "concat":
            lstm_input_size = 2 * 3 * self.cnn_out_channels  # 384
        elif self.fusion_type == "late":
            lstm_input_size = 3 * self.cnn_out_channels  # 192 (process separately)
        else:  # early or mid
            lstm_input_size = 3 * self.cnn_out_channels  # 192

        lstm_input = torch.randn(self.batch_size, self.seq_len, lstm_input_size)

        dual_lstm = DualPathLSTM(
            input_size=lstm_input_size,
            fast_hidden_size=self.lstm_hidden_size,
            slow_hidden_size=self.lstm_hidden_size,
            fast_num_layers=self.num_lstm_layers,
            slow_num_layers=self.num_lstm_layers
        )

        lstm_output, fast_hidden, slow_hidden = dual_lstm(lstm_input)

        # LSTM output shape: [batch, seq_len, hidden_size * 2] (bidirectional-like from dual paths)
        expected_lstm_shape = (self.batch_size, self.seq_len, self.lstm_hidden_size * 2)
        self.assertEqual(lstm_output.shape, expected_lstm_shape)

        # Check hidden state shapes
        # Each hidden state tuple is (h_n, c_n) for each layer
        expected_h_shape = (self.num_lstm_layers, self.batch_size, self.lstm_hidden_size)
        expected_c_shape = (self.num_lstm_layers, self.batch_size, self.lstm_hidden_size)

        self.assertEqual(fast_hidden[0].shape, expected_h_shape)
        self.assertEqual(fast_hidden[1].shape, expected_c_shape)
        self.assertEqual(slow_hidden[0].shape, expected_h_shape)
        self.assertEqual(slow_hidden[1].shape, expected_c_shape)

    def test_regression_head_shapes(self):
        """Test regression head output shapes."""
        # Input to regression head is concatenated hidden states from both LSTM paths
        lstm_output_size = self.lstm_hidden_size * 2  # 256

        regression_input = torch.randn(self.batch_size, lstm_output_size)

        regression_head = RegressionHead(
            input_size=lstm_output_size,
            hidden_sizes=[128, 64],
            output_size=1
        )

        output = regression_head(regression_input)

        # Output shape: [batch, 1]
        expected_shape = (self.batch_size, 1)
        self.assertEqual(output.shape, expected_shape)

        # Also test with temporal data (seq_len dimension)
        temporal_input = torch.randn(self.batch_size, self.seq_len, lstm_output_size)
        temporal_output = regression_head(temporal_input)

        # Output shape: [batch, seq_len, 1]
        expected_temporal_shape = (self.batch_size, self.seq_len, 1)
        self.assertEqual(temporal_output.shape, expected_temporal_shape)

    def test_full_model_shapes(self):
        """Test full RUL model output shapes."""
        model = RULModel(
            vib_in_channels=self.vib_in_channels,
            temp_in_channels=self.temp_in_channels,
            cnn_out_channels=self.cnn_out_channels,
            lstm_hidden_size=self.lstm_hidden_size,
            fusion_type=self.fusion_type,
            num_lstm_layers=self.num_lstm_layers,
            regression_hidden=[128, 64],
            dropout_rate=0.3
        )

        # Test forward pass
        output = model(self.vib_input, self.temp_input)

        # For sequence input, model should output predictions for each time step
        # Based on our implementation, it takes the last time step for final prediction
        # But let's check what our actual implementation does

        # Actually, looking at the RULModel.forward method, it takes the last time step
        # So output should be [batch_size, 1]
        expected_shape = (self.batch_size, 1)
        self.assertEqual(output.shape, expected_shape)

        # Test predict_rul method
        pred_output = model.predict_rul(self.vib_input, self.temp_input)
        self.assertEqual(pred_output.shape, expected_shape)

    def test_shape_consistency_with_documentation(self):
        """Test that shapes are consistent with values mentioned in MODEL_WORKFLOW_DETAILED.md."""
        # From documentation:
        # - CNN encoder outputs: 3 * 64 = 192 features
        # - LSTM hidden size: 128
        # - Regression head: 192 → 128 → 64 → 1 (but this seems to be for flattened features)
        # Actually, the documentation shows regression head as FC 192→128→64→1
        # This suggests it's operating on features after some pooling operation

        # Let's verify the CNN output dimension
        vib_encoder = VibrationEncoder(self.vib_in_channels, self.cnn_out_channels)
        vib_transposed = self.vib_input.transpose(1, 2)
        vib_features = vib_encoder(vib_transposed)

        # After global average pooling (as used in some baselines)
        vib_pooled = torch.mean(vib_features, dim=2)  # (batch, 3*out_channels)
        self.assertEqual(vib_pooled.shape[1], 3 * self.cnn_out_channels)
        self.assertEqual(vib_pooled.shape[1], 192)

        # This matches the "FC 192→128→64→1" mentioned in regression_head.py docstring

if __name__ == '__main__':
    unittest.main()