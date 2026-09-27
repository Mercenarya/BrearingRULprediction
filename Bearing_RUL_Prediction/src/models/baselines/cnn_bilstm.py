"""
Baseline model: CNN-BiLSTM for RUL prediction.
"""
import torch
import torch.nn as nn
from ..models.cnn_encoder import VibrationEncoder, TemperatureEncoder

class CNNBiLSTM(nn.Module):
    """
    CNN-BiLSTM baseline model.
    Uses CNN for feature extraction and BiLSTM for temporal modeling.
    """

    def __init__(self, vib_in_channels, temp_in_channels,
                 cnn_out_channels=64, lstm_hidden_size=128,
                 num_lstm_layers=2, regression_hidden=[128, 64],
                 dropout_rate=0.3):
        """
        Initialize the CNN-BiLSTM baseline.

        Args:
            vib_in_channels (int): Number of input vibration channels
            temp_in_channels (int): Number of input temperature channels
            cnn_out_channels (int): Output channels from CNN encoder
            lstm_hidden_size (int): Hidden size for LSTM layers
            num_lstm_layers (int): Number of LSTM layers
            regression_hidden (list): Hidden sizes for regression head
            dropout_rate (float): Dropout rate
        """
        super(CNNBiLSTM, self).__init__()

        # CNN Encoders for both modalities
        self.vib_encoder = VibrationEncoder(vib_in_channels, cnn_out_channels)
        self.temp_encoder = TemperatureEncoder(temp_in_channels, cnn_out_channels)

        # Feature dimensions after CNN encoding
        self.vib_feature_dim = 3 * cnn_out_channels
        self.temp_feature_dim = 3 * cnn_out_channels
        self.total_feature_dim = self.vib_feature_dim + self.temp_feature_dim

        # BiLSTM for temporal modeling
        self.bilstm = nn.LSTM(
            input_size=self.total_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0
        )

        # Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, regression_hidden[0]),  # *2 for bidirectional
            nn.BatchNorm1d(regression_hidden[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(regression_hidden[0], regression_hidden[1]),
            nn.BatchNorm1d(regression_hidden[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(regression_hidden[1], 1)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, vib_input, temp_input):
        """
        Forward pass through CNN-BiLSTM.

        Args:
            vib_input (torch.Tensor): Vibration input of shape
                                   (batch_size, seq_len, vib_in_channels)
            temp_input (torch.Tensor): Temperature input of shape
                                    (batch_size, seq_len, temp_in_channels)

        Returns:
            torch.Tensor: Predicted RUL values of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = vib_input.shape

        # CNN Encoding
        vib_transposed = vib_input.transpose(1, 2)
        temp_transposed = temp_input.transpose(1, 2)

        vib_features = self.vib_encoder(vib_transposed)
        temp_features = self.temp_encoder(temp_transposed)

        vib_features = vib_features.transpose(1, 2)
        temp_features = temp_features.transpose(1, 2)

        # Concatenate features from both modalities
        combined_features = torch.cat([vib_features, temp_features], dim=-1)
        combined_features = self.dropout(combined_features)

        # BiLSTM processing
        bilstm_output, _ = self.bilstm(combined_features)

        # Take the last time step output for prediction
        last_output = bilstm_output[:, -1, :]  # (batch_size, lstm_hidden_size * 2)

        # Regression head
        rul_prediction = self.regression_head(last_output)

        return rul_prediction

if __name__ == "__main__":
    # Example usage
    print("CNN-BiLSTM baseline ready")