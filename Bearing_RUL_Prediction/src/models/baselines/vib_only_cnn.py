"""
Baseline model: Vibration-only CNN for RUL prediction.
"""
import torch
import torch.nn as nn
from ..models.cnn_encoder import VibrationEncoder
from ..models.regression_head import RegressionHead

class VibOnlyCNN(nn.Module):
    """
    Vibration-only CNN baseline model.
    Uses only vibration data with CNN encoder and regression head.
    """

    def __init__(self, vib_in_channels, cnn_out_channels=64,
                 regression_hidden=[128, 64], dropout_rate=0.3):
        """
        Initialize the Vibration-only CNN baseline.

        Args:
            vib_in_channels (int): Number of input vibration channels
            cnn_out_channels (int): Output channels from CNN encoder
            regression_hidden (list): Hidden sizes for regression head
            dropout_rate (float): Dropout rate
        """
        super(VibOnlyCNN, self).__init__()

        # CNN Encoder for vibration only
        self.vib_encoder = VibrationEncoder(vib_in_channels, cnn_out_channels)
        self.feature_dim = 3 * cnn_out_channels  # Multi-scale features

        # Regression Head
        self.regression_head = RegressionHead(
            input_size=self.feature_dim,
            hidden_sizes=regression_hidden,
            output_size=1,
            dropout_rate=dropout_rate
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, vib_input):
        """
        Forward pass through vibration-only CNN.

        Args:
            vib_input (torch.Tensor): Vibration input of shape
                                   (batch_size, seq_len, vib_in_channels)

        Returns:
            torch.Tensor: Predicted RUL values of shape (batch_size, 1)
        """
        # CNN Encoding - extract multi-scale features
        vib_transposed = vib_input.transpose(1, 2)  # (batch, vib_in_channels, seq_len)
        vib_features = self.vib_encoder(vib_transposed)  # (batch, 3*out_channels, seq_len)
        vib_features = vib_features.transpose(1, 2)  # (batch, seq_len, 3*out_channels)

        # Global average pooling over time dimension
        vib_pooled = torch.mean(vib_features, dim=1)  # (batch_size, 3*out_channels)
        vib_pooled = self.dropout(vib_pooled)

        # Regression head
        rul_prediction = self.regression_head(vib_pooled)

        return rul_prediction

if __name__ == "__main__":
    # Example usage
    print("Vibration-only CNN baseline ready")