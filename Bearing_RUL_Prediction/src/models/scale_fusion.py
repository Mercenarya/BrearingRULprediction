"""
Scale fusion module for combining early, mid, late features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .common import BaseModel
from .config import ModelConfig


class ScaleFusion(BaseModel):
    """
    Fuse multi-scale features (early, mid, late) into a single tensor.

    Steps:
        1. Project each scale to d_model (if not already).
        2. Fusion mode:
            a. concat_project: concatenate the three scales and apply a linear projection.
            b. weighted_sum: project each scale to d_model, then compute weighted sum
               using softmax over scale weights (global or dynamic).

    Output: tensor of shape [B, T, d_model]
    If return_aux=True, also returns scale weights.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.fusion_type = config.fusion_type
        self.d_model = config.d_model
        self.scale_weight_type = config.scale_weight_type
        self.return_aux = config.return_aux

        # We'll create projection layers to map from input dimension to d_model.
        # The input dimension will be inferred from the input features in the forward pass.
        # We'll create the layers lazily in the forward pass or we can create them
        # here with a placeholder and recreate if needed.
        # For simplicity, we'll create them in the forward pass and cache them.

        # We'll implement two fusion modes:

        # 1. concat_project: concatenate the three scales along the feature dimension
        #    -> [B, T, 3 * d_model] then apply a linear projection to [B, T, d_model].
        # 2. weighted_sum: compute a weight for each scale (global or dynamic) and
        #    do a weighted sum of the three scales (each [B, T, d_model]) to get
        #    [B, T, d_model].

        # For weighted_sum, we need to compute weights:
        #   - Global: learnable parameters of shape [3] (softmax to get weights)
        #   - Dynamic: compute weights from the input, e.g., via a small network
        #        that outputs [B, T, 3] (then softmax over the last dimension).

        if self.fusion_type == "concat_project":
            self.concat_proj = nn.Linear(3 * self.d_model, self.d_model)
        elif self.fusion_type == "weighted_sum":
            if self.scale_weight_type == "global":
                # Global weights: one weight per scale
                self.scale_weights = nn.Parameter(torch.ones(3))  # We'll apply softmax
            elif self.scale_weight_type == "dynamic":
                # Dynamic weights: we'll use a small network to compute weights from
                # the concatenated features? Or from each scale? We'll use a simple
                # approach: apply a linear layer to each scale to get a score, then
                # softmax over scales.
                self.weight_net = nn.Linear(self.d_model, 1)  # Outputs [B*T, 1] per scale
            else:
                raise ValueError(f"Unsupported scale weight type: {self.scale_weight_type}")
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")

        # Cache for projection layers to avoid recreating them every forward pass
        self._proj_cache = {}

    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            features: dict with keys 'early', 'mid', 'late', each [B, T, D]
                    where D is the input feature dimension (may not equal d_model).

        Returns:
            Tuple of (fused_features, scale_aux) where:
                - fused_features: tensor of shape [B, T, d_model]
                - scale_aux: dict containing scale weights if return_aux=True, else empty dict
        """
        # We expect three scales
        early = features["early"]
        mid = features["mid"]
        late = features["late"]

        # Check shapes
        if early.shape != mid.shape or mid.shape != late.shape:
            raise ValueError("All scales must have the same shape")

        # Get the input feature dimension (should be the same for all scales)
        B, T, D_in = early.shape
        if mid.shape != (B, T, D_in) or late.shape != (B, T, D_in):
            raise ValueError("All scales must have the same shape [B, T, D_in]")

        # Get or create projection layers to map from D_in to d_model
        cache_key = D_in
        if cache_key not in self._proj_cache:
            self._proj_cache[cache_key] = nn.Linear(D_in, self.d_model).to(early.device)
        proj_layer = self._proj_cache[cache_key]

        # Project each scale to d_model
        early_proj = proj_layer(early)  # [B, T, d_model]
        mid_proj = proj_layer(mid)      # [B, T, d_model]
        late_proj = proj_layer(late)    # [B, T, d_model]

        scale_aux = {}
        if self.fusion_type == "concat_project":
            # Concatenate along the feature dimension
            concatenated = torch.cat([early_proj, mid_proj, late_proj], dim=-1)  # [B, T, 3 * d_model]
            fused = self.concat_proj(concatenated)  # [B, T, d_model]
        elif self.fusion_type == "weighted_sum":
            if self.scale_weight_type == "global":
                # Global weights: softmax to get [3]
                weights = F.softmax(self.scale_weights, dim=0)  # [3]
                # Reshape weights to [1, 1, 3] for broadcasting
                weights = weights.view(1, 1, 3)
                # Stack the scales to [B, T, 3, d_model]
                stacked = torch.stack([early_proj, mid_proj, late_proj], dim=-2)  # [B, T, 3, d_model]
                # Weighted sum: [B, T, d_model]
                fused = (stacked * weights.unsqueeze(-1)).sum(dim=-2)
                if self.return_aux:
                    scale_aux["scale_weights"] = weights  # [3]
            elif self.scale_weight_type == "dynamic":
                # For each scale, compute a score
                # We'll use the same weight_net for each scale? Or different?
                # We'll use the same network applied to each scale's features.
                # Flatten the batch and time dimensions
                B, T, D = early_proj.shape
                early_flat = early_proj.view(B * T, D)  # [B*T, D]
                mid_flat = mid_proj.view(B * T, D)
                late_flat = late_proj.view(B * T, D)

                # Compute scores for each scale
                early_score = self.weight_net(early_flat)  # [B*T, 1]
                mid_score = self.weight_net(mid_flat)      # [B*T, 1]
                late_score = self.weight_net(late_flat)    # [B*T, 1]

                # Concatenate scores: [B*T, 3]
                scores = torch.cat([early_score, mid_score, late_score], dim=-1)
                # Softmax over scales
                weights = F.softmax(scores, dim=-1)  # [B*T, 3]
                # Reshape to [B, T, 3]
                weights = weights.view(B, T, 3)

                # Compute weighted sum
                # Stack scales to [B, T, 3, D]
                stacked = torch.stack([early_proj, mid_proj, late_proj], dim=-2)  # [B, T, 3, D]
                # Weights to [B, T, 3, 1]
                weights = weights.unsqueeze(-1)
                fused = (stacked * weights).sum(dim=-2)  # [B, T, D]
                if self.return_aux:
                    scale_aux["scale_weights"] = weights  # [B, T, 3]
            else:
                raise ValueError(f"Unsupported scale weight type: {self.scale_weight_type}")
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")

        return fused, scale_aux