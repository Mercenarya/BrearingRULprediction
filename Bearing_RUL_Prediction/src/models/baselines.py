"""
Baseline models for RUL prediction.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .common import BaseModel
from .config import ModelConfig
from .vibration_encoder import VibrationEncoder
from .temperature_encoder import TemperatureEncoder
from .scale_fusion import ScaleFusion
from .temporal import TemporalEncoder
from .regression_head import RegressionHead


class VibrationCNNRegressor(BaseModel):
    """
    Vibration-only baseline: vibration encoder -> scale fusion -> temporal encoder -> regression head.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.vib_encoder = VibrationEncoder(config)
        self.scale_fusion = ScaleFusion(config)
        # Use standard LSTM for baselines (not dual-path)
        temporal_config = ModelConfig(
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            device=config.device,
            vib_in_channels=config.vib_in_channels,
            vib_lengths=config.vib_lengths,
            vib_cnn_channels=config.vib_cnn_channels,
            vib_kernel_sizes=config.vib_kernel_sizes,
            vib_pool_sizes=config.vib_pool_sizes,
            vib_norm_type=config.vib_norm_type,
            vib_activation=config.vib_activation,
            temp_representation=config.temp_representation,
            temp_in_features=config.temp_in_features,
            temp_history_length=config.temp_history_length,
            temp_encoder_type=config.temp_encoder_type,
            temp_hidden_sizes=config.temp_hidden_sizes,
            temp_norm_type=config.temp_norm_type,
            temp_activation=config.temp_activation,
            attn_num_heads=config.attn_num_heads,
            attn_dropout=config.attn_dropout,
            attn_proj_dropout=config.attn_proj_dropout,
            fusion_type=config.fusion_type,
            d_model=config.d_model,
            scale_weight_type=config.scale_weight_type,
            temporal_type="lstm",  # Use LSTM for baselines
            temporal_hidden_size=config.temporal_hidden_size,
            temporal_num_layers=config.temporal_num_layers,
            temporal_dropout=config.temporal_dropout,
            bidirectional=config.bidirectional,
            regression_hidden=config.regression_hidden,
            regression_dropout=config.regression_dropout,
            non_negative=config.non_negative,
            return_aux=config.return_aux
        )
        self.temporal_encoder = TemporalEncoder(temporal_config)
        self.regression_head = RegressionHead(config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        vib_input = batch["vibration"]
        # We don't use temperature in this baseline
        vib_features = self.vib_encoder(vib_input)  # dict of early, mid, late
        # Fuse the scales
        fused_features = self.scale_fusion(vib_features)  # [B, T, d_model]
        # Temporal encoding
        temporal_output, _, _ = self.temporal_encoder(fused_features)  # [B, T, d_model]
        # We assume the temporal encoder outputs [B, T, d_model] (config set appropriately)
        # Get a fixed-size representation per sequence for regression: last time step
        last_output = temporal_output[:, -1, :]  # [B, d_model]
        # Regression head
        rul_prediction = self.regression_head(last_output)  # [B, 1]

        return {
            "rul_pred": rul_prediction,
            "aux": {} if self.config.return_aux else None
        }


class TemperatureRegressor(BaseModel):
    """
    Temperature-only baseline: temperature encoder -> scale fusion -> temporal encoder -> regression head.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.temp_encoder = TemperatureEncoder(config)
        self.scale_fusion = ScaleFusion(config)
        # Use standard LSTM for baselines (not dual-path)
        temporal_config = ModelConfig(
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            device=config.device,
            vib_in_channels=config.vib_in_channels,
            vib_lengths=config.vib_lengths,
            vib_cnn_channels=config.vib_cnn_channels,
            vib_kernel_sizes=config.vib_kernel_sizes,
            vib_pool_sizes=config.vib_pool_sizes,
            vib_norm_type=config.vib_norm_type,
            vib_activation=config.vib_activation,
            temp_representation=config.temp_representation,
            temp_in_features=config.temp_in_features,
            temp_history_length=config.temp_history_length,
            temp_encoder_type=config.temp_encoder_type,
            temp_hidden_sizes=config.temp_hidden_sizes,
            temp_norm_type=config.temp_norm_type,
            temp_activation=config.temp_activation,
            attn_num_heads=config.attn_num_heads,
            attn_dropout=config.attn_dropout,
            attn_proj_dropout=config.attn_proj_dropout,
            fusion_type=config.fusion_type,
            d_model=config.d_model,
            scale_weight_type=config.scale_weight_type,
            temporal_type="lstm",  # Use LSTM for baselines
            temporal_hidden_size=config.temporal_hidden_size,
            temporal_num_layers=config.temporal_num_layers,
            temporal_dropout=config.temporal_dropout,
            bidirectional=config.bidirectional,
            regression_hidden=config.regression_hidden,
            regression_dropout=config.regression_dropout,
            non_negative=config.non_negative,
            return_aux=config.return_aux
        )
        self.temporal_encoder = TemporalEncoder(temporal_config)
        self.regression_head = RegressionHead(config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        temp_input = batch["temperature"]
        temp_mask = batch["temperature_mask"]
        temp_features = self.temp_encoder(temp_input, temp_mask)  # dict of early, mid, late
        fused_features = self.scale_fusion(temp_features)  # [B, T, d_model]
        temporal_output, _, _ = self.temporal_encoder(fused_features)  # [B, T, d_model]
        last_output = temporal_output[:, -1, :]  # [B, d_model]
        rul_prediction = self.regression_head(last_output)  # [B, 1]

        return {
            "rul_pred": rul_prediction,
            "aux": {} if self.config.return_aux else None
        }


class LateConcatFusionRegressor(BaseModel):
    """
    Late fusion baseline: encode each modality, fuse scales, concatenate modalities,
    -> temporal encoder -> regression head.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.vib_encoder = VibrationEncoder(config)
        self.temp_encoder = TemperatureEncoder(config)
        self.scale_fusion = ScaleFusion(config)
        # We'll fuse the modalities by concatenation after encoding each modality
        # We'll use a scale fusion for each modality to get [B, T, d_model] then
        # concatenate along the feature dimension -> [B, T, 2*d_model]
        # Then we'll project to d_model? Or we can let the temporal encoder
        # handle the increased input size? We'll change the temporal encoder
        # to accept an input size of 2*d_model? We'll add a projection.

        # We'll do: encode vibration and temperature, fuse scales for each,
        # then concatenate the fused features -> [B, T, 2*d_model]
        # Then we'll apply a linear projection to get [B, T, d_model] before
        # the temporal encoder.

        # Use standard LSTM for baselines (not dual-path)
        temporal_config = ModelConfig(
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            device=config.device,
            vib_in_channels=config.vib_in_channels,
            vib_lengths=config.vib_lengths,
            vib_cnn_channels=config.vib_cnn_channels,
            vib_kernel_sizes=config.vib_kernel_sizes,
            vib_pool_sizes=config.vib_pool_sizes,
            vib_norm_type=config.vib_norm_type,
            vib_activation=config.vib_activation,
            temp_representation=config.temp_representation,
            temp_in_features=config.temp_in_features,
            temp_history_length=config.temp_history_length,
            temp_encoder_type=config.temp_encoder_type,
            temp_hidden_sizes=config.temp_hidden_sizes,
            temp_norm_type=config.temp_norm_type,
            temp_activation=config.temp_activation,
            attn_num_heads=config.attn_num_heads,
            attn_dropout=config.attn_dropout,
            attn_proj_dropout=config.attn_proj_dropout,
            fusion_type=config.fusion_type,
            d_model=config.d_model,
            scale_weight_type=config.scale_weight_type,
            temporal_type="lstm",  # Use LSTM for baselines
            temporal_hidden_size=config.temporal_hidden_size,
            temporal_num_layers=config.temporal_num_layers,
            temporal_dropout=config.temporal_dropout,
            bidirectional=config.bidirectional,
            regression_hidden=config.regression_hidden,
            regression_dropout=config.regression_dropout,
            non_negative=config.non_negative,
            return_aux=config.return_aux
        )
        self.modality_fusion = nn.Linear(2 * config.d_model, config.d_model)
        self.temporal_encoder = TemporalEncoder(temporal_config)
        self.regression_head = RegressionHead(config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        vib_input = batch["vibration"]
        temp_input = batch["temperature"]
        temp_mask = batch["temperature_mask"]

        vib_features = self.vib_encoder(vib_input)  # dict of early, mid, late
        temp_features = self.temp_encoder(temp_input, temp_mask)  # dict of early, mid, late

        # Fuse scales for each modality
        vib_fused = self.scale_fusion(vib_features)  # [B, T, d_model]
        temp_fused = self.scale_fusion(temp_features)  # [B, T, d_model]

        # Concatenate modalities
        concatenated = torch.cat([vib_fused, temp_fused], dim=-1)  # [B, T, 2*d_model]
        # Project to d_model
        fused_features = self.modality_fusion(concatenated)  # [B, T, d_model]

        # Temporal encoding
        temporal_output, _, _ = self.temporal_encoder(fused_features)  # [B, T, d_model]
        last_output = temporal_output[:, -1, :]  # [B, d_model]
        rul_prediction = self.regression_head(last_output)  # [B, 1]

        return {
            "rul_pred": rul_prediction,
            "aux": {} if self.config.return_aux else None
        }


class CNNTemporalRegressor(BaseModel):
    """
    CNN temporal baseline: encode vibration and temperature, concatenate scales,
    concatenate modalities, project to d_model, -> temporal encoder -> regression head.
    The plan says: "concat hai modality rồi GRU/LSTM"
    We interpret as: concatenate the vibration and temperature modalities (after
    encoding and concatenating scales) and then apply a temporal encoder (GRU/LSTM).
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.vib_encoder = VibrationEncoder(config)
        self.temp_encoder = TemperatureEncoder(config)
        # We need the feature dimensions from the encoders
        self.vib_feature_dim = config.vibration_feature_dim
        self.temp_feature_dim = config.temperature_feature_dim
        # We'll concatenate the three scales for each modality, then concatenate modalities
        # Input to modality_fusion: 3 * (vib_feature_dim + temp_feature_dim)
        self.modality_fusion = nn.Linear(
            3 * (self.vib_feature_dim + self.temp_feature_dim),
            config.d_model
        )
        # Use standard LSTM for baselines (not dual-path)
        temporal_config = ModelConfig(
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            device=config.device,
            vib_in_channels=config.vib_in_channels,
            vib_lengths=config.vib_lengths,
            vib_cnn_channels=config.vib_cnn_channels,
            vib_kernel_sizes=config.vib_kernel_sizes,
            vib_pool_sizes=config.vib_pool_sizes,
            vib_norm_type=config.vib_norm_type,
            vib_activation=config.vib_activation,
            temp_representation=config.temp_representation,
            temp_in_features=config.temp_in_features,
            temp_history_length=config.temp_history_length,
            temp_encoder_type=config.temp_encoder_type,
            temp_hidden_sizes=config.temp_hidden_sizes,
            temp_norm_type=config.temp_norm_type,
            temp_activation=config.temp_activation,
            attn_num_heads=config.attn_num_heads,
            attn_dropout=config.attn_dropout,
            attn_proj_dropout=config.attn_proj_dropout,
            fusion_type=config.fusion_type,
            d_model=config.d_model,
            scale_weight_type=config.scale_weight_type,
            temporal_type="lstm",  # Use LSTM for baselines
            temporal_hidden_size=config.temporal_hidden_size,
            temporal_num_layers=config.temporal_num_layers,
            temporal_dropout=config.temporal_dropout,
            bidirectional=config.bidirectional,
            regression_hidden=config.regression_hidden,
            regression_dropout=config.regression_dropout,
            non_negative=config.non_negative,
            return_aux=config.return_aux
        )
        self.temporal_encoder = TemporalEncoder(temporal_config)
        self.regression_head = RegressionHead(config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        vib_input = batch["vibration"]
        temp_input = batch["temperature"]
        temp_mask = batch["temperature_mask"]

        # Encode vibration and temperature
        vib_features = self.vib_encoder(vib_input)  # dict of early, mid, late
        temp_features = self.temp_encoder(temp_input, temp_mask)  # dict of early, mid, late

        # For each modality, concatenate the three scales
        vib_concat = torch.cat([
            vib_features["early"],
            vib_features["mid"],
            vib_features["late"]
        ], dim=-1)  # [B, T, 3 * vib_feature_dim]
        temp_concat = torch.cat([
            temp_features["early"],
            temp_features["mid"],
            temp_features["late"]
        ], dim=-1)  # [B, T, 3 * temp_feature_dim]

        # Concatenate vibration and temperature modalities
        concatenated = torch.cat([vib_concat, temp_concat], dim=-1)  # [B, T, 3*(vib_feature_dim+temp_feature_dim)]
        # Project to d_model
        fused_features = self.modality_fusion(concatenated)  # [B, T, d_model]

        # Temporal encoding
        temporal_output, _, _ = self.temporal_encoder(fused_features)  # [B, T, d_model]
        last_output = temporal_output[:, -1, :]  # [B, d_model]
        rul_prediction = self.regression_head(last_output)  # [B, 1]

        return {
            "rul_pred": rul_prediction,
            "aux": {} if self.config.return_aux else None
        }


class SingleScaleCrossAttentionRegressor(BaseModel):
    """
    Model to isolate the contribution of cross-attention.
    Uses a single scale (mid) from vibration and temperature encoders,
    applies cross-modal attention, combines the attended features,
    then uses a temporal encoder and regression head.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.vib_encoder = VibrationEncoder(config)
        self.temp_encoder = TemperatureEncoder(config)
        self.cross_modal_attn = CrossModalAttention(config)
        # After cross-attention, we have features of size d_model for each modality
        # We'll combine the vibration and temperature attended features by concatenation and projection
        self.modality_proj = nn.Linear(2 * config.d_model, config.d_model)
        # Use standard LSTM for this ablation (not dual-path)
        temporal_config = ModelConfig(
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            device=config.device,
            vib_in_channels=config.vib_in_channels,
            vib_lengths=config.vib_lengths,
            vib_cnn_channels=config.vib_cnn_channels,
            vib_kernel_sizes=config.vib_kernel_sizes,
            vib_pool_sizes=config.vib_pool_sizes,
            vib_norm_type=config.vib_norm_type,
            vib_activation=config.vib_activation,
            temp_representation=config.temp_representation,
            temp_in_features=config.temp_in_features,
            temp_history_length=config.temp_history_length,
            temp_encoder_type=config.temp_encoder_type,
            temp_hidden_sizes=config.temp_hidden_sizes,
            temp_norm_type=config.temp_norm_type,
            temp_activation=config.temp_activation,
            attn_num_heads=config.attn_num_heads,
            attn_dropout=config.attn_dropout,
            attn_proj_dropout=config.attn_proj_dropout,
            fusion_type=config.fusion_type,
            d_model=config.d_model,
            scale_weight_type=config.scale_weight_type,
            temporal_type="lstm",  # Use LSTM for ablation
            temporal_hidden_size=config.temporal_hidden_size,
            temporal_num_layers=config.temporal_num_layers,
            temporal_dropout=config.temporal_dropout,
            bidirectional=config.bidirectional,
            regression_hidden=config.regression_hidden,
            regression_dropout=config.regression_dropout,
            non_negative=config.non_negative,
            return_aux=config.return_aux
        )
        self.temporal_encoder = TemporalEncoder(temporal_config)
        self.regression_head = RegressionHead(config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        vib_input = batch["vibration"]
        temp_input = batch["temperature"]
        temp_mask = batch["temperature_mask"]

        # Step 1: Encode vibration and temperature to get multi-scale features
        vib_features = self.vib_encoder(vib_input)  # dict of early, mid, late
        temp_features = self.temp_encoder(temp_input, temp_mask)  # dict of early, mid, late

        # Step 2: Select the mid scale for each modality
        vib_mid = vib_features["mid"]  # [B, T, D_v]
        temp_mid = temp_features["mid"]  # [B, T, D_t]

        # Step 3: Cross-modal attention on the mid scale
        # We need to format the features as dictionaries for the cross_attention module
        vib_features_dict = {"mid": vib_mid}
        temp_features_dict = {"mid": temp_mid}
        vib_attended, temp_attended = self.cross_modal_attn(
            vib_features_dict, temp_features_dict, temperature_mask=temp_mask
        )
        vib_mid_attended = vib_attended["mid"]  # [B, T, d_model]
        temp_mid_attended = temp_attended["mid"]  # [B, T, d_model]

        # Step 4: Combine the attended features from vibration and temperature
        concatenated = torch.cat([vib_mid_attended, temp_mid_attended], dim=-1)  # [B, T, 2*d_model]
        fused_features = self.modality_proj(concatenated)  # [B, T, d_model]

        # Step 5: Temporal encoding
        temporal_output, _, _ = self.temporal_encoder(fused_features)  # [B, T, d_model]
        last_output = temporal_output[:, -1, :]  # [B, d_model]

        # Step 6: Regression head
        rul_prediction = self.regression_head(last_output)  # [B, 1]

        return {
            "rul_pred": rul_prediction,
            "aux": {} if self.config.return_aux else None
        }


class MultiScaleFusionRegressor(BaseModel):
    """
    Model to isolate the contribution of multi-scale fusion.
    Encodes vibration and temperature, fuses the scales for each modality,
    combines the modalities, then uses a temporal encoder and regression head.
    (No cross-attention)
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.vib_encoder = VibrationEncoder(config)
        self.temp_encoder = TemperatureEncoder(config)
        self.scale_fusion = ScaleFusion(config)
        # We'll fuse the modalities by concatenation after encoding each modality
        # We'll use a scale fusion for each modality to get [B, T, d_model] then
        # concatenate along the feature dimension -> [B, T, 2*d_model]
        # Then we'll project to d_model before the temporal encoder.
        self.modality_fusion = nn.Linear(2 * config.d_model, config.d_model)
        # Use standard LSTM for this ablation (not dual-path)
        temporal_config = ModelConfig(
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            device=config.device,
            vib_in_channels=config.vib_in_channels,
            vib_lengths=config.vib_lengths,
            vib_cnn_channels=config.vib_cnn_channels,
            vib_kernel_sizes=config.vib_kernel_sizes,
            vib_pool_sizes=config.vib_pool_sizes,
            vib_norm_type=config.vib_norm_type,
            vib_activation=config.vib_activation,
            temp_representation=config.temp_representation,
            temp_in_features=config.temp_in_features,
            temp_history_length=config.temp_history_length,
            temp_encoder_type=config.temp_encoder_type,
            temp_hidden_sizes=config.temp_hidden_sizes,
            temp_norm_type=config.temp_norm_type,
            temp_activation=config.temp_activation,
            attn_num_heads=config.attn_num_heads,
            attn_dropout=config.attn_dropout,
            attn_proj_dropout=config.attn_proj_dropout,
            fusion_type=config.fusion_type,
            d_model=config.d_model,
            scale_weight_type=config.scale_weight_type,
            temporal_type="lstm",  # Use LSTM for ablation
            temporal_hidden_size=config.temporal_hidden_size,
            temporal_num_layers=config.temporal_num_layers,
            temporal_dropout=config.temporal_dropout,
            bidirectional=config.bidirectional,
            regression_hidden=config.regression_hidden,
            regression_dropout=config.regression_dropout,
            non_negative=config.non_negative,
            return_aux=config.return_aux
        )
        self.temporal_encoder = TemporalEncoder(temporal_config)
        self.regression_head = RegressionHead(config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        vib_input = batch["vibration"]
        temp_input = batch["temperature"]
        temp_mask = batch["temperature_mask"]

        # Step 1: Encode vibration and temperature to get multi-scale features
        vib_features = self.vib_encoder(vib_input)  # dict of early, mid, late
        temp_features = self.temp_encoder(temp_input, temp_mask)  # dict of early, mid, late

        # Step 2: Fuse scales for each modality
        vib_fused = self.scale_fusion(vib_features)  # [B, T, d_model]
        temp_fused = self.scale_fusion(temp_features)  # [B, T, d_model]

        # Step 3: Combine modalities
        concatenated = torch.cat([vib_fused, temp_fused], dim=-1)  # [B, T, 2*d_model]
        fused_features = self.modality_fusion(concatenated)  # [B, T, d_model]

        # Step 4: Temporal encoding
        temporal_output, _, _ = self.temporal_encoder(fused_features)  # [B, T, d_model]
        last_output = temporal_output[:, -1, :]  # [B, d_model]

        # Step 5: Regression head
        rul_prediction = self.regression_head(last_output)  # [B, 1]

        return {
            "rul_pred": rul_prediction,
            "aux": {} if self.config.return_aux else None
        }


class MultiScaleCrossAttentionRegressor(BaseModel):
    """
    Model that combines multi-scale fusion and cross-modal attention
    but uses a simple temporal encoder (not dual-path).
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.vib_encoder = VibrationEncoder(config)
        self.temp_encoder = TemperatureEncoder(config)
        self.cross_modal_attn = CrossModalAttention(config)
        self.scale_fusion = ScaleFusion(config)
        # We'll combine the vibration and temperature features after cross-attention and scale fusion
        # by concatenation and projection
        self.modality_proj = nn.Linear(2 * config.d_model, config.d_model)
        # Use standard LSTM for this ablation (not dual-path)
        temporal_config = ModelConfig(
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            device=config.device,
            vib_in_channels=config.vib_in_channels,
            vib_lengths=config.vib_lengths,
            vib_cnn_channels=config.vib_cnn_channels,
            vib_kernel_sizes=config.vib_kernel_sizes,
            vib_pool_sizes=config.vib_pool_sizes,
            vib_norm_type=config.vib_norm_type,
            vib_activation=config.vib_activation,
            temp_representation=config.temp_representation,
            temp_in_features=config.temp_in_features,
            temp_history_length=config.temp_history_length,
            temp_encoder_type=config.temp_encoder_type,
            temp_hidden_sizes=config.temp_hidden_sizes,
            temp_norm_type=config.temp_norm_type,
            temp_activation=config.temp_activation,
            attn_num_heads=config.attn_num_heads,
            attn_dropout=config.attn_dropout,
            attn_proj_dropout=config.attn_proj_dropout,
            fusion_type=config.fusion_type,
            d_model=config.d_model,
            scale_weight_type=config.scale_weight_type,
            temporal_type="lstm",  # Use LSTM for ablation
            temporal_hidden_size=config.temporal_hidden_size,
            temporal_num_layers=config.temporal_num_layers,
            temporal_dropout=config.temporal_dropout,
            bidirectional=config.bidirectional,
            regression_hidden=config.regression_hidden,
            regression_dropout=config.regression_dropout,
            non_negative=config.non_negative,
            return_aux=config.return_aux
        )
        self.temporal_encoder = TemporalEncoder(temporal_config)
        self.regression_head = RegressionHead(config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        vib_input = batch["vibration"]
        temp_input = batch["temperature"]
        temp_mask = batch["temperature_mask"]

        # Step 1: Encode vibration and temperature to get multi-scale features
        vib_features = self.vib_encoder(vib_input)  # dict of early, mid, late
        temp_features = self.temp_encoder(temp_input, temp_mask)  # dict of early, mid, late

        # Step 2: Cross-modal attention
        vib_attended, temp_attended = self.cross_modal_attn(
            vib_features, temp_features, temperature_mask=temp_mask
        )  # each is dict of early, mid, late

        # Step 3: For each scale, combine vibration and temperature features
        fused_per_scale = {}
        for scale in ['early', 'mid', 'late']:
            vib_scale = vib_attended[scale]  # [B, T, d_model]
            temp_scale = temp_attended[scale]  # [B, T, d_model]
            # Concatenate along feature dimension
            concatenated = torch.cat([vib_scale, temp_scale], dim=-1)  # [B, T, 2*d_model]
            # Project to d_model
            projected = self.modality_proj(concatenated)  # [B, T, d_model]
            fused_per_scale[scale] = projected

        # Step 4: Scale fusion
        fused_features = self.scale_fusion(fused_per_scale)  # [B, T, d_model]

        # Step 5: Temporal encoding
        temporal_output, _, _ = self.temporal_encoder(fused_features)  # [B, T, d_model]
        last_output = temporal_output[:, -1, :]  # [B, d_model]

        # Step 6: Regression head
        rul_prediction = self.regression_head(last_output)  # [B, 1]

        return {
            "rul_pred": rul_prediction,
            "aux": {} if self.config.return_aux else None
        }