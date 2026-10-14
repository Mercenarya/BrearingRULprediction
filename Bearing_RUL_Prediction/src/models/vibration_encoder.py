"""
Vibration encoder for multi-scale feature extraction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from .common import BaseModel, get_activation, get_normalization
from .config import ModelConfig


class VibrationEncoder(BaseModel):
    """
    Vibration encoder that processes [B, T, 2, L] and returns multi-scale features.

    Steps:
        1. Reshape to [B*T, 2, L]
        2. Apply shared 1D CNN blocks with normalization and activation
        3. Multi-scale pooled embeddings
        4. Restore [B, T, D]

    Output: dictionary with keys 'early', 'mid', 'late', each [B, T, D_scale]
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.vib_in_channels = config.vib_in_channels
        self.vib_cnn_channels = config.vib_cnn_channels  # List of three integers
        self.vib_kernel_sizes = config.vib_kernel_sizes
        self.vib_pool_sizes = config.vib_pool_sizes
        self.vib_norm_type = config.vib_norm_type
        self.vib_activation = config.vib_activation
        self.vibration_feature_dim = config.vibration_feature_dim  # Target dimension for each scale

        # Build the shared CNN blocks
        self.blocks = nn.ModuleList()
        in_channels = self.vib_in_channels
        for out_channels, kernel_size, pool_size in zip(
            self.vib_cnn_channels, self.vib_kernel_sizes, self.vib_pool_sizes
        ):
            block = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                get_normalization(self.vib_norm_type, out_channels),
                get_activation(self.vib_activation),
                nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
            )
            self.blocks.append(block)
            in_channels = out_channels

        # Projection layers to map each block's output to the target vibration_feature_dim
        self.scale_projections = nn.ModuleList()
        for out_channels in self.vib_cnn_channels:
            self.scale_projections.append(
                nn.Linear(out_channels, self.vibration_feature_dim)
            )

        # Compute and store receptive field and parameter count for logging
        self._receptive_field = self._compute_receptive_field()
        self._parameter_count = self._count_parameters()

    def _compute_receptive_field(self) -> int:
        """Compute the receptive field of the vibration encoder."""
        receptive_field = 1
        for kernel_size, pool_size in zip(self.vib_kernel_sizes, self.vib_pool_sizes):
            receptive_field = (receptive_field - 1) * pool_size + kernel_size
        return receptive_field

    def _count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_receptive_field(self) -> int:
        """Return the receptive field of the encoder."""
        return self._receptive_field

    def get_parameter_count(self) -> int:
        """Return the number of trainable parameters."""
        return self._parameter_count

    def forward(self, vibration: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            vibration: [B, T, 2, L_vib]

        Returns:
            Dictionary with keys 'early', 'mid', 'late', each [B, T, vibration_feature_dim]
        """
        B, T, _, L = vibration.shape
        # Reshape to [B*T, 2, L]
        x = vibration.view(B * T, self.vib_in_channels, L)

        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)  # [B*T, C_out, L_out]
            # Apply adaptive average pooling to get [B*T, C_out, 1]
            x_pooled = F.adaptive_avg_pool1d(x, output_size=1)
            x_pooled = x_pooled.squeeze(-1)  # [B*T, C_out]
            # Reshape to [B, T, C_out]
            x_pooled = x_pooled.view(B, T, -1)  # [B, T, C_out]
            # Project to vibration_feature_dim
            x_pooled = self.scale_projections[i](x_pooled)  # [B, T, vibration_feature_dim]
            features.append(x_pooled)

        # We expect three scales: early, mid, late
        if len(features) != 3:
            raise ValueError(f"Expected 3 scales from vibration encoder, got {len(features)}")

        return {
            "early": features[0],
            "mid": features[1],
            "late": features[2],
        }