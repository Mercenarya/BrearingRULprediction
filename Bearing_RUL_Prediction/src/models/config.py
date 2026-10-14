"""
Model configuration for Bearing RUL prediction.
"""
from dataclasses import dataclass, field
from typing import Optional, Literal, Union, List, Dict, Any
import torch


@dataclass
class ModelConfig:
    """Configuration for all models."""
    # Common parameters
    batch_size: int = 4
    sequence_length: int = 32
    device: str = "cpu"

    # Vibration encoder parameters
    vib_in_channels: int = 2  # X and Y axes
    vib_lengths: List[int] = field(default_factory=lambda: [1024, 2560])  # Supported lengths
    vib_cnn_channels: List[int] = field(default_factory=lambda: [64, 64, 64])  # For early, mid, late scales
    vib_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    vib_pool_sizes: List[int] = field(default_factory=lambda: [2, 2, 2])
    vib_norm_type: Literal["batch", "layer", "none"] = "batch"
    vib_activation: Literal["relu", "gelu", "elu"] = "relu"

    # Temperature encoder parameters
    temp_representation: Literal["feature_vector", "history"] = "feature_vector"
    temp_in_features: int = 7  # For feature vector
    temp_history_length: int = 64  # For history representation
    temp_encoder_type: Literal["mlp", "conv1d", "temporal_mlp"] = "mlp"
    temp_hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    temp_norm_type: Literal["batch", "layer", "none"] = "batch"
    temp_activation: Literal["relu", "gelu", "elu"] = "relu"

    # Cross-modal attention parameters
    attn_num_heads: int = 4
    attn_dropout: float = 0.1
    attn_proj_dropout: float = 0.1

    # Scale fusion parameters
    fusion_type: Literal["concat_project", "weighted_sum"] = "weighted_sum"
    d_model: int = 128  # Dimension after projection for each scale
    scale_weight_type: Literal["global", "dynamic"] = "global"

    # Temporal encoder parameters
    temporal_type: Literal["gru", "lstm", "dual_path"] = "dual_path"
    temporal_hidden_size: int = 128
    temporal_num_layers: int = 2
    temporal_dropout: float = 0.1
    bidirectional: bool = False  # Only for single GRU/LSTM, not dual_path

    # Regression head parameters
    regression_hidden: List[int] = field(default_factory=lambda: [128, 64])
    regression_dropout: float = 0.1
    non_negative: bool = True  # Use Softplus or ReLU to ensure non-negative output

    # Auxiliary output
    return_aux: bool = False

    # Derived properties (set after initialization)
    vibration_feature_dim: int = field(init=False)
    temperature_feature_dim: int = field(init=False)

    def __post_init__(self):
        # Compute feature dimensions from encoders
        # For vibration: sum of channels from CNN blocks (early, mid, late)
        self.vibration_feature_dim = sum(self.vib_cnn_channels)
        # For temperature: depends on representation and encoder
        if self.temp_representation == "feature_vector":
            self.temperature_feature_dim = self.temp_hidden_sizes[-1] if self.temp_hidden_sizes else self.temp_in_features
        else:  # history
            # For history, we assume the encoder outputs a feature vector of size temp_hidden_sizes[-1]
            self.temperature_feature_dim = self.temp_hidden_sizes[-1] if self.temp_hidden_sizes else self.temp_history_length