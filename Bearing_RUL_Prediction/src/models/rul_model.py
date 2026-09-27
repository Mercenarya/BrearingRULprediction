"""
End-to-end RUL prediction model combining all components:
- CNN Encoder (multi-scale)
- Cross-modal Attention
- Dual-Path LSTM
- Regression Head
"""
import torch
import torch.nn as nn
from .cnn_encoder import VibrationEncoder, TemperatureEncoder
from .cross_modal_attention import CrossModalAttention, FusionStrategy
from .temporal import DualPathLSTM
from .regression_head import RegressionHead

class RULModel(nn.Module):
    """
    End-to-end model for Bearing Remaining Useful Life prediction.
    """

    def __init__(self, vib_in_channels, temp_in_channels,
                 cnn_out_channels=64, lstm_hidden_size=128,
                 fusion_type="concat", num_lstm_layers=2,
                 regression_hidden=[128, 64], dropout_rate=0.3):
        """
        Initialize the RUL Model.

        Args:
            vib_in_channels (int): Number of input vibration channels
            temp_in_channels (int): Number of input temperature channels
            cnn_out_channels (int): Output channels from CNN encoder
            lstm_hidden_size (int): Hidden size for LSTM layers
            fusion_type (str): Type of fusion strategy
            num_lstm_layers (int): Number of LSTM layers
            regression_hidden (list): Hidden sizes for regression head
            dropout_rate (float): Dropout rate
        """
        super(RULModel, self).__init__()

        # CNN Encoders for each modality
        self.vib_encoder = VibrationEncoder(vib_in_channels, cnn_out_channels)
        self.temp_encoder = TemperatureEncoder(temp_in_channels, cnn_out_channels)

        # After CNN encoding, we have 3 * cnn_out_channels features from multi-scale
        self.vib_feature_dim = 3 * cnn_out_channels
        self.temp_feature_dim = 3 * cnn_out_channels

        # Cross-modal attention
        self.cross_modal_attn = CrossModalAttention(
            feature_dim=self.vib_feature_dim,
            num_heads=4
        )

        # Fusion strategy
        self.fusion = FusionStrategy(
            vib_feature_dim=self.vib_feature_dim,
            temp_feature_dim=self.temp_feature_dim,
            fusion_type=fusion_type
        )

        # Determine LSTM input size based on fusion type
        if fusion_type == "late":
            # For late fusion, we process modalities separately
            self.lstm_input_size = self.vib_feature_dim
        else:
            # For early/mid/concat fusion, we have fused features
            if fusion_type == "concat":
                self.lstm_input_size = self.vib_feature_dim + self.temp_feature_dim
            else:
                # early or mid fusion outputs vib_feature_dim (same as input after fusion)
                self.lstm_input_size = self.vib_feature_dim

        # Dual-Path LSTM
        self.temporal_model = DualPathLSTM(
            input_size=self.lstm_input_size,
            fast_hidden_size=lstm_hidden_size,
            slow_hidden_size=lstm_hidden_size,
            fast_num_layers=num_lstm_layers,
            slow_num_layers=num_lstm_layers,
            dropout=dropout_rate
        )

        # Regression Head
        self.regression_head = RegressionHead(
            input_size=lstm_hidden_size * 2,  # fast + slow hidden sizes
            hidden_sizes=regression_hidden,
            output_size=1,
            dropout_rate=dropout_rate
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, vib_input, temp_input):
        """
        Forward pass through the complete RUL model.

        Args:
            vib_input (torch.Tensor): Vibration input of shape
                                   (batch_size, seq_len, vib_in_channels)
            temp_input (torch.Tensor): Temperature input of shape
                                    (batch_size, seq_len, temp_in_channels)

        Returns:
            torch.Tensor: Predicted RUL values of shape (batch_size, 1)
                         or (batch_size, seq_len, 1) for sequence output
        """
        batch_size, seq_len, _ = vib_input.shape

        # CNN Encoding - extract multi-scale features
        # Note: CNN expects (batch, channels, seq_len) so we need to transpose
        vib_transposed = vib_input.transpose(1, 2)  # (batch, vib_in_channels, seq_len)
        temp_transposed = temp_input.transpose(1, 2)  # (batch, temp_in_channels, seq_len)

        vib_features = self.vib_encoder(vib_transposed)  # (batch, 3*out_channels, seq_len)
        temp_features = self.temp_encoder(temp_transposed)  # (batch, 3*out_channels, seq_len)

        # Transpose back to (batch, seq_len, features) for attention and LSTM
        vib_features = vib_features.transpose(1, 2)  # (batch, seq_len, 3*out_channels)
        temp_features = temp_features.transpose(1, 2)  # (batch, seq_len, 3*out_channels)

        # Cross-modal attention
        vib_attended, temp_attended = self.cross_modal_attn(vib_features, temp_features)

        # Fusion
        if self.fusion.fusion_type == "late":
            # For late fusion, keep modalities separate for LSTM processing
            vib_lstm_out, (vib_fast_h, vib_fast_c), (vib_slow_h, vib_slow_c) = \
                self.temporal_model(vib_attended)
            temp_lstm_out, (temp_fast_h, temp_fast_c), (temp_slow_h, temp_slow_c) = \
                self.temporal_model(temp_attended)

            # Combine LSTM outputs from both modalities
            lstm_output = torch.cat([vib_lstm_out, temp_lstm_out], dim=-1)
        else:
            # For early/mid/concat fusion, fuse features first
            fused_features = self.fusion(vib_attended, temp_attended)
            fused_features = self.dropout(fused_features)

            # Process through temporal model
            lstm_output, _, _ = self.temporal_model(fused_features)

        # Extract final hidden states for regression
        # For sequence output, we could take the last time step
        if len(lstm_output.shape) == 3:
            # Take the last time step output for RUL prediction
            lstm_features = lstm_output[:, -1, :]  # (batch_size, lstm_hidden_size * 2)
        else:
            lstm_features = lstm_output

        # Regression head
        rul_prediction = self.regression_head(lstm_features)

        return rul_prediction

    def predict_rul(self, vib_input, temp_input):
        """
        Predict RUL from vibration and temperature inputs.

        Args:
            vib_input (torch.Tensor): Vibration input
            temp_input (torch.Tensor): Temperature input

        Returns:
            torch.Tensor: Predicted RUL values
        """
        self.eval()
        with torch.no_grad():
            return self.forward(vib_input, temp_input)

if __name__ == "__main__":
    # Example usage
    print("RUL Model module ready")