"""
Full multimodal RUL model with dual-path temporal encoder.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .common import BaseModel
from .config import ModelConfig
from .vibration_encoder import VibrationEncoder
from .temperature_encoder import TemperatureEncoder
from .cross_attention import CrossModalAttention
from .scale_fusion import ScaleFusion
from .temporal import DualPathTemporalEncoder
from .regression_head import RegressionHead


class MultimodalRUL(BaseModel):
    """
    Full model for RUL prediction with dual-path temporal encoder.

    Architecture:
        Vibration input -> Vibration Encoder (multi-scale) ->
        Temperature input -> Temperature Encoder (multi-scale) ->
        Cross-modal attention (vibration and temperature) ->
        Scale fusion (early, mid, late) ->
        Dual-path temporal encoder ->
        Regression head ->
        RUL prediction
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

        # Encoders
        self.vib_encoder = VibrationEncoder(config)
        self.temp_encoder = TemperatureEncoder(config)

        # Cross-modal attention
        self.cross_modal_attn = CrossModalAttention(config)

        # Scale fusion
        self.scale_fusion = ScaleFusion(config)

        # Modality projection: combine vibration and temperature features
        self.modality_proj = nn.Linear(2 * config.d_model, config.d_model)

        # Dual-path temporal encoder
        self.temporal_encoder = DualPathTemporalEncoder(config)

        # Regression head
        self.regression_head = RegressionHead(config)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: dict with keys:
                - vibration: [B, T, 2, L_vib]
                - temperature: [B, T, F_temp] or [B, T, 1, 64]
                - temperature_mask: [B, T, F_temp] or [B, T, 1, 64] (bool)
                - rul: [B, 1] (not used in forward)

        Returns:
            dict with:
                - rul_pred: [B, 1]
                - aux: dict (optional) containing:
                    * scale_weights: [B, T, 3] or [3] if return_aux=True
                    * cross_attention: dict of attention weights if return_aux=True
                    * temporal_weights: dict of temporal weights if return_aux=True
        """
        vib_input = batch["vibration"]
        temp_input = batch["temperature"]
        temp_mask = batch["temperature_mask"]

        # Step 1: Encode vibration and temperature to get multi-scale features
        vib_features = self.vib_encoder(vib_input)  # dict of early, mid, late
        temp_features = self.temp_encoder(temp_input, temp_mask)  # dict of early, mid, late

        # Step 2: Cross-modal attention
        vib_attended, temp_attended, attn_aux = self.cross_modal_attn(
            vib_features, temp_features, temperature_mask=temp_mask
        )  # each is dict of early, mid, late; attn_aux contains attention weights if return_aux=True

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
        fused_features, scale_aux = self.scale_fusion(fused_per_scale)  # fused_features: [B, T, d_model]; scale_aux contains scale weights if return_aux=True

        # Step 5: Dual-path temporal encoding
        temporal_output, _, _, temp_aux = self.temporal_encoder(fused_features)  # [B, T, d_model]; temp_aux contains temporal weights if return_aux=True

        # Step 6: Get a fixed-size representation per sequence for regression
        # We'll use the last time step
        last_output = temporal_output[:, -1, :]  # [B, d_model]

        # Step 7: Regression head
        rul_prediction = self.regression_head(last_output)  # [B, 1]

        # Prepare auxiliary outputs
        aux = {}
        if self.config.return_aux:
            # Combine aux from all components
            aux.update(attn_aux)
            aux.update(scale_aux)
            aux.update(temp_aux)

        return {
            "rul_pred": rul_prediction,
            "aux": aux if self.config.return_aux else None
        }