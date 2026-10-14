"""
End-to-end RUL prediction model combining all components:
- CNN Encoder (multi-scale)
- Cross-modal Attention
- Dual-Path Temporal Encoder
- Regression Head
"""
import torch
import torch.nn as nn
from .vibration_encoder import VibrationEncoder
from .temperature_encoder import TemperatureEncoder
from .cross_attention import CrossModalAttention
from .scale_fusion import ScaleFusion
from .temporal import DualPathTemporalEncoder
from .regression_head import RegressionHead
from .config import ModelConfig


class RULModel(nn.Module):
    """
    End-to-end model for Bearing Remaining Useful Life prediction.
    """

    def __init__(self, vib_in_channels=2, temp_in_channels=1,
                 cnn_out_channels=64, lstm_hidden_size=128,
                 fusion_type="concat", num_lstm_layers=2,
                 regression_hidden=[128, 64], dropout_rate=0.3):
        """
        Initialize the RUL Model.
        """
        super(RULModel, self).__init__()

        # Store the fusion type for modality combination (after scale fusion)
        self.modality_fusion_type = fusion_type

        # Create a ModelConfig object from the individual parameters
        self.config = ModelConfig(
            # Vibration encoder parameters
            vib_in_channels=vib_in_channels,
            vib_cnn_channels=[cnn_out_channels, cnn_out_channels, cnn_out_channels],  # Three scales: early, mid, late

            # Temperature encoder parameters
            temp_in_features=1,                   # Dummy value for feature vector (not used in history mode)
            temp_representation="history",        # Use history representation for raw temperature signal
            temp_history_length=1024,             # Will match window size from data
            temp_encoder_type="mlp",              # Using MLP for temperature history encoding
            temp_hidden_sizes=[64, 64],           # Hidden layers for temp encoder history

            # Model architecture parameters
            fusion_type="concat_project",         # Use concat_project for internal scale fusion
            d_model=cnn_out_channels * 3,         # After CNN encoding, we have 3 * cnn_out_channels features per modality

            # Temporal encoder parameters
            temporal_hidden_size=lstm_hidden_size,
            temporal_num_layers=num_lstm_layers,

            # Regression head parameters
            regression_hidden=regression_hidden,
            regression_dropout=dropout_rate,
            non_negative=True
        )

        # CNN Encoders for each modality
        self.vib_encoder = VibrationEncoder(self.config)
        self.temp_encoder = TemperatureEncoder(self.config)

        # After CNN encoding, we get the feature dimensions from the config
        self.vib_feature_dim = self.config.vibration_feature_dim
        self.temp_feature_dim = self.config.temperature_feature_dim

        # Cross-modal attention
        self.cross_modal_attn = CrossModalAttention(self.config)

        # Scale fusion (using concat_project internally for combining early, mid, late)
        self.vib_scale_fusion = ScaleFusion(self.config)
        self.temp_scale_fusion = ScaleFusion(self.config)

        # Modality projection: combine vibration and temperature features
        # The modality_fusion_type will determine how we combine them
        if self.modality_fusion_type == "concat":
            # Concatenate vibration and temperature features, then project
            self.modality_proj = nn.Linear(2 * self.config.d_model, self.config.d_model)
        elif self.modality_fusion_type == "early" or self.modality_fusion_type == "mid" or self.modality_fusion_type == "late":
            # For early/mid/late fusion, we would select just one scale
            # But since we've already done scale fusion, we'll just use the fused features directly
            # and project from single modality dimension to d_model
            self.modality_proj = nn.Linear(self.config.d_model, self.config.d_model)
        else:
            # Default to concat projection
            self.modality_proj = nn.Linear(2 * self.config.d_model, self.config.d_model)

        # Dual-path temporal encoder
        self.temporal_encoder = DualPathTemporalEncoder(self.config)

        # Regression Head
        self.regression_head = RegressionHead(self.config)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, vib_input, temp_input):
        """
        Forward pass through the complete RUL model.
        """
        batch_size = vib_input.shape[0]

        # Step 0: Reshape input data to match encoder expectations
        # From DataLoader we get:
        #   vib_input: [batch_size, window_size, 2]  # [N, L, C] where L=window_size, C=channels
        #   temp_input: [batch_size, window_size, 1]  # [N, L, C] where L=window_size, C=channels
        #
        # Encoders expect:
        #   vibration: [B, T, 2, L_vib] where B=batch, T=sequence length, 2=channels, L=window length
        #   temperature (history): [B, T, 1, L_temp] where B=batch, T=sequence length, 1=channel, L_temp=window length
        #
        # Since we want to treat each window as an independent time step (no temporal sequencing yet),
        # we set T=1 and reshape accordingly:
        #
        # For vibration: [batch_size, window_size, 2] -> [batch_size, 1, 2, window_size]
        # For temperature: [batch_size, window_size, 1] -> [batch_size, 1, 1, window_size]
        #
        # NOTE: The vibration encoder expects channels to come BEFORE window length in the last two dimensions:
        # [B, T, channels, window_length]
        # NOTE: The temperature encoder expects [B, T, 1, history_length] for history representation

        # Reshape vibration input: [batch_size, window_size, 2] -> [batch_size, 1, 2, window_size]
        vib_input_reshaped = vib_input.unsqueeze(1).transpose(-1, -2)  # [B, 1, 2, L]

        # Reshape temperature input: [batch_size, window_size, 1] -> [batch_size, 1, 1, window_size]
        temp_input_reshaped = temp_input.unsqueeze(1).transpose(-1, -2)  # [B, 1, 1, L]

        # Step 1: Encode vibration and temperature to get multi-scale features
        vib_features = self.vib_encoder(vib_input_reshaped)  # dict of early, mid, late [B, T, d_model]
        temp_features = self.temp_encoder(temp_input_reshaped)  # dict of early, mid, late [B, T, d_model]

        # Step 2: Cross-modal attention
        vib_attended, temp_attended, _ = self.cross_modal_attn(
            vib_features, temp_features
        )  # each is dict of early, mid, late, third return is attention weights (unused)

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

        # Step 4: Scale fusion (combine early, mid, late features for each modality)
        vib_fused, _ = self.vib_scale_fusion(fused_per_scale)  # [B, T, d_model]
        temp_fused, _ = self.temp_scale_fusion(fused_per_scale)  # [B, T, d_model]

        # Step 5: Combine vibration and temperature features based on modality_fusion_type
        if self.modality_fusion_type == "concat":
            # Concatenate vibration and temperature features
            combined_modalities = torch.cat([vib_fused, temp_fused], dim=-1)  # [B, T, 2*d_model]
            # Project back to d_model for the temporal encoder
            combined_proj = nn.Linear(2 * self.config.d_model, self.config.d_model).to(vib_input.device)
            fused_features = combined_proj(combined_modalities)  # [B, T, d_model]
        elif self.modality_fusion_type == "early":
            # Use early scale features only
            fused_features = vib_fused  # Already [B, T, d_model] from scale fusion
        elif self.modality_fusion_type == "mid":
            # Use mid scale features only
            fused_features = temp_fused  # Already [B, T, d_model] from scale fusion
        elif self.modality_fusion_type == "late":
            # Use late scale features only
            # Actually, we should pick one consistently - let's use vibration for late
            fused_features = vib_fused  # Already [B, T, d_model] from scale fusion
        else:
            # Default to concat
            combined_modalities = torch.cat([vib_fused, temp_fused], dim=-1)  # [B, T, 2*d_model]
            combined_proj = nn.Linear(2 * self.config.d_model, self.config.d_model).to(vib_input.device)
            fused_features = combined_proj(combined_modalities)  # [B, T, d_model]

        # Step 6: Dual-path temporal encoding
        temporal_output, _, _, _ = self.temporal_encoder(fused_features)  # [B, T, d_model]

        # Step 7: Get a fixed-size representation per sequence for regression
        # We'll use the last time step
        last_output = temporal_output[:, -1, :]  # [B, d_model]

        # Step 8: Regression head
        rul_prediction = self.regression_head(last_output)  # [B, 1]

        return rul_prediction

    def predict_rul(self, vib_input, temp_input):
        self.eval()
        with torch.no_grad():
            return self.forward(vib_input, temp_input)

if __name__ == "__main__":
    print("RUL Model module ready")