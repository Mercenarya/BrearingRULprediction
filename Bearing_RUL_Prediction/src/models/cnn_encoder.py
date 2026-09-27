"""
CNN Encoder with 3-scale feature extraction (early/mid/late) for vibration and temperature data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    CNN Encoder that extracts features at multiple scales.
    """

    def __init__(self, in_channels, out_channels=64, kernel_sizes=[3, 5, 7]):
        """
        Initialize the CNN Encoder.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels per scale
            kernel_sizes (list): List of kernel sizes for early, mid, late layers
        """
        super(CNNEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        # Early scale features (small kernel)
        self.early_conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0]//2
        )
        self.early_bn = nn.BatchNorm1d(out_channels)

        # Mid scale features (medium kernel)
        self.mid_conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1]//2
        )
        self.mid_bn = nn.BatchNorm1d(out_channels)

        # Late scale features (large kernel)
        self.late_conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_sizes[2],
            padding=kernel_sizes[2]//2
        )
        self.late_bn = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass through the CNN encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length)

        Returns:
            torch.Tensor: Concatenated features from all scales of shape
                         (batch_size, 3*out_channels, sequence_length)
        """
        # Early scale
        early = self.relu(self.early_bn(self.early_conv(x)))

        # Mid scale
        mid = self.relu(self.mid_bn(self.mid_conv(x)))

        # Late scale
        late = self.relu(self.late_bn(self.late_conv(x)))

        # Concatenate features from all scales
        multi_scale_features = torch.cat([early, mid, late], dim=1)
        return multi_scale_features

class VibrationEncoder(CNNEncoder):
    """CNN Encoder specifically for vibration data."""
    def __init__(self, in_channels, out_channels=64):
        super(VibrationEncoder, self).__init__(in_channels, out_channels)

class TemperatureEncoder(CNNEncoder):
    """CNN Encoder specifically for temperature data."""
    def __init__(self, in_channels, out_channels=64):
        super(TemperatureEncoder, self).__init__(in_channels, out_channels)

if __name__ == "__main__":
    # Example usage
    print("CNN Encoder module ready")