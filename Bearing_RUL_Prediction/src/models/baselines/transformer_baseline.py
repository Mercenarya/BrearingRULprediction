"""
Baseline model: Transformer for RUL prediction.
"""
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional encoding.

        Args:
            d_model (int): Model dimension
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass for positional encoding.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Input with positional encoding added
        """
        return x + self.pe[:x.size(0), :]

class TransformerBaseline(nn.Module):
    """
    Transformer baseline model for RUL prediction.
    """

    def __init__(self, vib_in_channels, temp_in_channels,
                 d_model=128, nhead=8, num_encoder_layers=3,
                 dim_feedforward=512, dropout=0.1):
        """
        Initialize the Transformer baseline.

        Args:
            vib_in_channels (int): Number of input vibration channels
            temp_in_channels (int): Number of input temperature channels
            d_model (int): Model dimension
            nhead (int): Number of attention heads
            num_encoder_layers (int): Number of transformer encoder layers
            dim_feedforward (int): Dimension of feedforward network
            dropout (float): Dropout rate
        """
        super(TransformerBaseline, self).__init__()
        self.d_model = d_model

        # Input projection layers
        self.vib_proj = nn.Linear(vib_in_channels, d_model)
        self.temp_proj = nn.Linear(temp_in_channels, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_encoder_layers
        )

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, vib_input, temp_input):
        """
        Forward pass through Transformer baseline.

        Args:
            vib_input (torch.Tensor): Vibration input of shape
                                   (batch_size, seq_len, vib_in_channels)
            temp_input (torch.Tensor): Temperature input of shape
                                    (batch_size, seq_len, temp_in_channels)

        Returns:
            torch.Tensor: Predicted RUL values of shape (batch_size, 1)
        """
        # Project inputs to model dimension
        vib_proj = self.vib_proj(vib_input)  # (batch, seq_len, d_model)
        temp_proj = self.temp_proj(temp_input)  # (batch, seq_len, d_model)

        # Combine modalities (simple concatenation and projection)
        combined = torch.cat([vib_proj, temp_proj], dim=-1)  # (batch, seq_len, 2*d_model)
        combined = nn.Linear(2*self.d_model, self.d_model)(combined)  # Project back to d_model

        # Add positional encoding
        combined = combined.transpose(0, 1)  # (seq_len, batch, d_model) for transformer
        combined = self.pos_encoder(combined)
        combined = combined.transpose(0, 1)  # (batch, seq_len, d_model)

        # Apply dropout
        combined = self.dropout(combined)

        # Transformer encoding
        transformer_output = self.transformer_encoder(combined)  # (batch, seq_len, d_model)

        # Global average pooling
        pooled_output = torch.mean(transformer_output, dim=1)  # (batch, d_model)

        # Regression head
        rul_prediction = self.regression_head(pooled_output)

        return rul_prediction

if __name__ == "__main__":
    # Example usage
    print("Transformer baseline ready")