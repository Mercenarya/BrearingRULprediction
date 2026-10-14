"""
Cross-modal attention block for vibration and temperature features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .common import BaseModel
from .config import ModelConfig


class CrossModalAttention(BaseModel):
    """
    Cross-modal attention that runs on temporal tokens within each sample.

    Steps:
        1. Normalize dimension before attention: project vibration and temperature
           scales to d_model.
        2. Cross-attention in two directions:
            a. Vibration query, temperature key/value.
            b. Temperature query, vibration key/value.
        3. Combine the two results (e.g., concatenate and project, or add).
        4. Residual connection, normalization, and dropout.

    Output: tuple (vib_attended, temp_attended), each being a dict with
            keys 'early', 'mid', 'late', each [B, T, d_model]
    If return_aux=True, also returns attention weights.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.d_model = config.d_model
        self.num_heads = config.attn_num_heads
        self.dropout = config.attn_dropout
        self.proj_dropout = config.attn_proj_dropout
        self.return_aux = config.return_aux

        # We assume the input features from vibration and temperature encoders
        # are of dimensions: vibration_feature_dim and temperature_feature_dim
        # (which we get from the config after post_init)
        self.vib_feature_dim = config.vibration_feature_dim
        self.temp_feature_dim = config.temperature_feature_dim

        # Linear projections to d_model for query, key, value
        self.vib_proj_q = nn.Linear(self.vib_feature_dim, self.d_model)
        self.vib_proj_k = nn.Linear(self.vib_feature_dim, self.d_model)
        self.vib_proj_v = nn.Linear(self.vib_feature_dim, self.d_model)

        self.temp_proj_q = nn.Linear(self.temp_feature_dim, self.d_model)
        self.temp_proj_k = nn.Linear(self.temp_feature_dim, self.d_model)
        self.temp_proj_v = nn.Linear(self.temp_feature_dim, self.d_model)

        # Multi-head attention layers
        self.vib_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,  # We want [B, T, D] format
        )
        self.temp_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )

        # Output projections
        self.vib_out_proj = nn.Linear(self.d_model, self.d_model)
        self.temp_out_proj = nn.Linear(self.d_model, self.d_model)

        # Residual connections and normalization
        self.vib_norm = nn.LayerNorm(self.d_model)
        self.temp_norm = nn.LayerNorm(self.d_model)
        self.vib_dropout = nn.Dropout(self.proj_dropout)
        self.temp_dropout = nn.Dropout(self.proj_dropout)

    def forward(
        self,
        vib_features: Dict[str, torch.Tensor],
        temp_features: Dict[str, torch.Tensor],
        temperature_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            vib_features: dict with keys 'early', 'mid', 'late', each [B, T, D_v]
            temp_features: dict with keys 'early', 'mid', 'late', each [B, T, D_t]
            temperature_mask: [B, T, F_temp] or [B, T, 1, 64] bool mask for temperature

        Returns:
            Tuple of (vib_attended, temp_attended, attn_aux) where:
                - vib_attended: dict with keys 'early', 'mid', 'late', each [B, T, d_model]
                - temp_attended: dict with keys 'early', 'mid', 'late', each [B, T, d_model]
                - attn_aux: dict containing attention weights if return_aux=True, else empty dict
        """
        # We need to combine the multi-scale features into a single tensor per modality?
        # The plan says: "Chuẩn hóa dimension trước attention:"
        #   vibration scale [B, T, D_v]
        #   temperature scale [B, T, D_t]
        #   -> linear projection về d_model
        #   -> cross-attention trong từng sample
        #   -> residual + normalization
        #   -> fused scale [B, T, d_model]

        # This suggests we do attention per scale? Or we combine the scales first?
        # The plan does not specify. We'll assume we do attention per scale independently
        # and then we return attended features per scale.

        # However, the plan also says: "Attention phải chạy trên temporal tokens trong từng sample,
        # Không attention qua chiều batch." which we are doing.

        # We'll process each scale separately: early, mid, late.

        vib_attended = {}
        temp_attended = {}
        attn_aux = {}

        # Prepare the temperature mask for attention: we need to convert it to
        # an attention mask of shape [B, T] (or [B, 1, T]?) for the key padding mask.
        # The MultiheadAttention in PyTorch with batch_first=True expects a key_padding_mask
        # of shape [B, T] where T is the sequence length of the key.
        # We have temperature_mask of shape [B, T, F_temp] or [B, T, 1, 64].
        # We'll reduce it to [B, T] by taking any over the feature dimensions? Actually,
        # we want to mask a whole time step if any feature is missing? Or we want to
        # mask per feature? The plan says: "Temperature mask phải được chuyển đúng thành
        # attention mask. Không tạo ma trận attention giữa các phần tử của batch."
        # We'll create a mask of shape [B, T] where True means the token is valid (not masked).
        # We'll define: a time step is valid if all features are not masked? Actually, the
        # mask we have is True for valid data (as in the synthetic data generation).
        if temperature_mask is not None:
            # Reduce the mask to [B, T] by checking if any feature is valid? Actually, we
            # want to mask the time step if all features are invalid? We'll do:
            #   valid_mask = temperature_mask.any(dim=-1)   # [B, T] if any feature is valid
            # But note: the mask might be of shape [B, T, 1, 64] for history.
            if temperature_mask.dim() == 4:  # [B, T, 1, 64]
                # Reduce over the last two dimensions
                valid_mask = temperature_mask.any(dim=(-1, -2))  # [B, T]
            elif temperature_mask.dim() == 3:  # [B, T, F_temp]
                valid_mask = temperature_mask.any(dim=-1)  # [B, T]
            else:
                raise ValueError(f"Unexpected temperature_mask shape: {temperature_mask.shape}")
            # Now, valid_mask is [B, T] with True for valid (we want to keep) and False for masked.
            # The key_padding_mask in MultiheadAttention expects True for positions to be masked.
            # So we invert: key_padding_mask = ~valid_mask
            key_padding_mask = ~valid_mask
        else:
            key_padding_mask = None

        # Process each scale
        for scale in ['early', 'mid', 'late']:
            vib_scale = vib_features[scale]  # [B, T, D_v]
            temp_scale = temp_features[scale]  # [B, T, D_t]

            # Project to d_model
            vib_proj = self.vib_proj_q(vib_scale)  # [B, T, d_model]
            temp_proj = self.temp_proj_q(temp_scale)  # [B, T, d_model]

            # For key and value, we use the projected values as well
            vib_k = self.vib_proj_k(vib_scale)  # [B, T, d_model]
            vib_v = self.vib_proj_v(vib_scale)  # [B, T, d_model]
            temp_k = self.temp_proj_k(temp_scale)  # [B, T, d_model]
            temp_v = self.temp_proj_v(temp_scale)  # [B, T, d_model]

            # Cross-attention: vibration query, temperature key/value
            vib_attended_raw, vib_attn_weights = self.vib_attn(
                query=vib_proj,
                key=temp_k,
                value=temp_v,
                key_padding_mask=key_padding_mask,
                need_weights=self.return_aux
            )
            # Cross-attention: temperature query, vibration key/value
            temp_attended_raw, temp_attn_weights = self.temp_attn(
                query=temp_proj,
                key=vib_k,
                value=vib_v,
                key_padding_mask=key_padding_mask,
                need_weights=self.return_aux
            )

            # Output projection
            vib_attended_raw = self.vib_out_proj(vib_attended_raw)  # [B, T, d_model]
            temp_attended_raw = self.temp_out_proj(temp_attended_raw)  # [B, T, d_model]

            # Residual connection, normalization, dropout
            vib_attended[scale] = self.vib_norm(vib_proj + self.vib_dropout(vib_attended_raw))
            temp_attended[scale] = self.temp_norm(temp_proj + self.temp_dropout(temp_attended_raw))

            # Store attention weights if needed
            if self.return_aux:
                # We'll store the attention weights for this scale
                # vib_attn_weights and temp_attn_weights are of shape [B, num_heads, T, T]
                attn_aux[f'vib_query_temp_key_{scale}'] = vib_attn_weights
                attn_aux[f'temp_query_vib_key_{scale}'] = temp_attn_weights

        return vib_attended, temp_attended, attn_aux