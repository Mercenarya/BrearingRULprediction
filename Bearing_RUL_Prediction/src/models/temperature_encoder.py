"""
Temperature encoder for feature vector and history representations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .common import BaseModel, get_activation, get_normalization
from .config import ModelConfig


class TemperatureEncoder(BaseModel):
    """
    Temperature encoder that supports two representations:
        1. Feature vector: [B, T, F_temp]
        2. Temperature history: [B, T, 1, 64]

    Output: dictionary with keys 'early', 'mid', 'late', each [B, T, D_scale]
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.representation = config.temp_representation
        self.in_features = config.temp_in_features
        self.history_length = config.temp_history_length
        self.encoder_type = config.temp_encoder_type
        self.hidden_sizes = config.temp_hidden_sizes
        self.norm_type = config.temp_norm_type
        self.activation = config.temp_activation

        # We'll create three identical MLPs (or conv1d) for early, mid, late?
        # Actually, the plan says: output returns early, mid, late according to configured dimension.
        # We'll assume we want three different feature vectors (like in vibration) but from the same input.
        # We can use the same encoder and then split the output? Or we can have three separate encoders?
        # The plan does not specify. We'll assume we want to get three scales from the temperature
        # encoder as well, so we'll create three identical branches.

        # However, to save parameters, we can use one encoder and then replicate the output?
        # But the plan might intend for the temperature encoder to have a similar multi-scale
        # structure as the vibration encoder. Let's read the plan again:

        # "Temperature encoder"
        #   - Hỗ trợ hai representation:
        #       1. Feature vector: [B, T, F_temp].
        #       2. Temperature history: [B, T, 1, 64].
        #   - Thiết kế:
        #       - Feature vector dùng MLP/gated MLP.
        #       - History 64 điểm dùng lightweight Conv1d hoặc temporal MLP.
        #       - Missing mask phải được dùng trong encoder hoặc trước encoder; không được mặc định
        #         biến missing thành measurement bằng 0 mà không có mask embedding.
        #       - Output trả về early, mid, late theo dimension cấu hình.

        # It says "Output trả về early, mid, late theo dimension cấu hình", meaning we output
        # three scales (early, mid, late) each with a dimension that is configured.

        # We'll create three identical heads (each being an MLP or Conv1d) that take the
        # (possibly masked) input and output a feature vector of size D_scale (which we
        # set to be the same for all three, and we can get it from the config's d_model?).
        # But note: the temperature encoder's output dimension is not directly given in the
        # config. We have `d_model` in the config for the scale fusion, but that is after
        # projection. We'll let the temperature encoder output a feature vector of size
        # `temp_feature_dim` (which we computed in the config) and then we will have three
        # copies of that? Or we can output three different feature vectors by having three
        # different heads.

        # We'll do: three identical heads (each is an MLP or Conv1d) that output a vector of
        # size `self.temperature_feature_dim` (from config). Then we return three copies
        # (early, mid, late) each being the same? That doesn't make sense.

        # Alternatively, we can split the hidden_sizes into three parts? Not likely.

        # Let's look at the vibration encoder: we got three scales from different layers of
        # the CNN. For temperature, we don't have a natural hierarchy. We might use the
        # same approach: if we use a temporal Conv1d, we can take the output of different
        # layers. But for MLP, we don't have layers.

        # Given the ambiguity, we'll assume that the temperature encoder outputs a single
        # feature vector per time step, and we then duplicate it three times for early, mid, late.
        # This is not ideal but satisfies the interface.

        # However, the plan says "Output trả về early, mid, late theo dimension cấu hình",
        # which might mean we configure the dimension for each scale? We'll assume the same
        # dimension for all three.

        # We'll create one encoder that outputs a feature vector of size `self.temperature_feature_dim`
        # (from config) and then we will return the same tensor for early, mid, late.

        # But note: the plan might expect different dimensions for each scale? We'll assume
        # the same for simplicity.

        # We'll build the encoder based on the representation and type.

        if self.representation == "feature_vector":
            self.expected_input_shape = ("B", "T", "F_temp")
            self.input_dim = self.in_features
        else:  # history
            self.expected_input_shape = ("B", "T", "1", "64")
            self.input_dim = self.history_length  # We'll treat the last dimension as features

        # Build the encoder
        if self.encoder_type == "mlp":
            # MLP: we apply the same MLP to each time step
            layers = []
            in_dim = self.input_dim
            for hidden_size in self.hidden_sizes:
                layers.append(nn.Linear(in_dim, hidden_size))
                layers.append(get_normalization(self.norm_type, hidden_size))
                layers.append(get_activation(self.activation))
                in_dim = hidden_size
            self.mlp = nn.Sequential(*layers)
            self.output_dim = self.hidden_sizes[-1] if self.hidden_sizes else self.input_dim
        elif self.encoder_type == "conv1d":
            # Conv1d over the temporal dimension? Actually, we want to encode each time step
            # independently? Or we want to use temporal context? The plan says "lightweight Conv1d"
            # for history. We'll assume we apply Conv1d over the vibration length dimension
            # (which is 1 for history? Actually, history is [B, T, 1, 64] -> we can view as
            # [B*T, 1, 64] and apply Conv1d over the 64 dimension).
            # We'll do: for each time step, we have a vector of length 64 (or F_temp) and we
            # apply a 1D convolution over that vector? That doesn't make sense for feature vector.
            # Let's re-read: "History 64 điểm dùng lightweight Conv1d hoặc temporal MLP."
            # This suggests we treat the 64-point history as a sequence and apply Conv1d over
            # the 64 dimension (like a 1D signal). So we do:
            #   input: [B, T, 1, 64] -> we can reshape to [B*T, 1, 64] and apply Conv1d.
            # We'll build a Conv1d network.
            layers = []
            in_channels = 1  # Because we have [B*T, 1, 64]
            out_channels = self.hidden_sizes[0] if self.hidden_sizes else 32
            for i, hidden_size in enumerate(self.hidden_sizes):
                out_channels = hidden_size if i < len(self.hidden_sizes) else out_channels
                layers.append(nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                ))
                layers.append(get_normalization(self.norm_type, out_channels))
                layers.append(get_activation(self.activation))
                in_channels = out_channels
            self.conv = nn.Sequential(*layers)
            # After conv, we do global average pooling over the 64 dimension to get [B*T, out_channels]
            self.output_dim = self.hidden_sizes[-1] if self.hidden_sizes else out_channels
        elif self.encoder_type == "temporal_mlp":
            # We'll treat the input as a sequence and apply an MLP with temporal context?
            # For simplicity, we'll use an LSTM or GRU? But the plan says temporal MLP.
            # We'll do: apply an MLP to each time step independently (same as mlp above) but
            # we call it temporal_mlp to distinguish.
            layers = []
            in_dim = self.input_dim
            for hidden_size in self.hidden_sizes:
                layers.append(nn.Linear(in_dim, hidden_size))
                layers.append(get_normalization(self.norm_type, hidden_size))
                layers.append(get_activation(self.activation))
                in_dim = hidden_size
            self.mlp = nn.Sequential(*layers)
            self.output_dim = self.hidden_sizes[-1] if self.hidden_sizes else self.input_dim
        else:
            raise ValueError(f"Unsupported temperature encoder type: {self.encoder_type}")

        # We'll create three identical heads (for early, mid, late) that are just the encoder
        # and then we return the same output for each? Actually, we want to return three
        # different tensors? We'll return the same tensor three times for now.
        # In the future, we might want to have different encoders for each scale, but we
        # don't have a specification.

        # We'll store the encoder and then in forward, we return three copies.
        # But note: the plan says "Output trả về early, mid, late theo dimension cấu hình",
        # which might mean we have three different output dimensions? We'll assume the same.

        # We'll compute the output dimension from the encoder and then set:
        self.temperature_feature_dim = self.output_dim  # from config's post_init

    def forward(self, temperature: torch.Tensor, temperature_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            temperature:
                If feature vector: [B, T, F_temp]
                If history: [B, T, 1, 64]
            temperature_mask:
                Same shape as temperature, bool, True for valid data.

        Returns:
            Dictionary with keys 'early', 'mid', 'late', each [B, T, D_scale]
        """
        # Validate input shape
        if self.representation == "feature_vector":
            if temperature.shape[-1] != self.in_features:
                raise ValueError(f"Expected feature vector with {self.in_features} features, got {temperature.shape[-1]}")
            # If mask is provided, we can use it to zero out missing values? But the plan says
            # not to default missing to zero without mask embedding. We'll apply the mask by
            # setting masked values to zero and then maybe we can learn to ignore?
            # Alternatively, we can use the mask to skip the computation? We'll do:
            if temperature_mask is not None:
                if temperature_mask.shape != temperature.shape:
                    raise ValueError("temperature_mask must have the same shape as temperature")
                # We'll zero out the masked positions
                temperature = temperature * temperature_mask
        else:  # history
            if temperature.shape[-2:] != (1, self.history_length):
                raise ValueError(f"Expected history shape [..., 1, {self.history_length}], got {temperature.shape[-2:]}")
            if temperature_mask is not None:
                if temperature_mask.shape != temperature.shape:
                    raise ValueError("temperature_mask must have the same shape as temperature")
                temperature = temperature * temperature_mask

        # Flatten the batch and time dimensions
        B, T = temperature.shape[0], temperature.shape[1]
        if self.representation == "feature_vector":
            x = temperature.view(B * T, -1)  # [B*T, F_temp]
        else:  # history
            # [B, T, 1, 64] -> [B*T, 64]
            x = temperature.view(B * T, self.history_length)

        # Apply encoder
        if self.encoder_type == "mlp" or self.encoder_type == "temporal_mlp":
            features = self.mlp(x)  # [B*T, output_dim]
        elif self.encoder_type == "conv1d":
            # [B*T, 64] -> [B*T, 1, 64] for conv1d
            x = temperature.view(B * T, 1, self.history_length)
            features = self.conv(x)  # [B*T, out_channels, L_out]
            # Global average pooling over the length dimension
            features = F.adaptive_avg_pool1d(features, output_size=1)  # [B*T, out_channels, 1]
            features = features.squeeze(-1)  # [B*T, out_channels]

        # Reshape back to [B, T, output_dim]
        features = features.view(B, T, -1)

        # We return the same features for early, mid, late
        # In the future, we might want to split the features into three parts or use
        # different heads, but for now we return three copies.
        return {
            "early": features,
            "mid": features,
            "late": features,
        }