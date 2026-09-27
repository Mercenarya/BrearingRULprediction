"""
Cross-modal attention mechanism for fusing vibration and temperature features.
Implements Early/Mid/Late fusion strategies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing features from two modalities.
    """

    def __init__(self, feature_dim, num_heads=4):
        """
        Initialize the Cross-modal Attention module.

        Args:
            feature_dim (int): Dimension of input features
            num_heads (int): Number of attention heads
        """
        super(CrossModalAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections for both modalities
        self.vib_q = nn.Linear(feature_dim, feature_dim)
        self.vib_k = nn.Linear(feature_dim, feature_dim)
        self.vib_v = nn.Linear(feature_dim, feature_dim)

        self.temp_q = nn.Linear(feature_dim, feature_dim)
        self.temp_k = nn.Linear(feature_dim, feature_dim)
        self.temp_v = nn.Linear(feature_dim, feature_dim)

        self.output_vib = nn.Linear(feature_dim, feature_dim)
        self.output_temp = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, vib_features, temp_features):
        """
        Forward pass through cross-modal attention.

        Args:
            vib_features (torch.Tensor): Vibration features of shape (batch, seq_len, feature_dim)
            temp_features (torch.Tensor): Temperature features of shape (batch, seq_len, feature_dim)

        Returns:
            tuple: (attended_vib, attended_temp) - features after cross-modal attention
        """
        batch_size, seq_len, _ = vib_features.shape

        # Vibration to Temperature attention
        vib_q = self.vib_q(vib_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        temp_k = self.temp_k(temp_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        temp_v = self.temp_v(temp_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Temperature to Vibration attention
        temp_q = self.temp_q(temp_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        vib_k = self.vib_k(vib_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        vib_v = self.vib_v(vib_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        vib_to_temp_scores = torch.matmul(vib_q, temp_k.transpose(-2, -1)) * self.scale
        temp_to_vib_scores = torch.matmul(temp_q, vib_k.transpose(-2, -1)) * self.scale

        vib_to_temp_attn = F.softmax(vib_to_temp_scores, dim=-1)
        temp_to_vib_attn = F.softmax(temp_to_vib_scores, dim=-1)

        # Apply dropout to attention weights
        vib_to_temp_attn = self.dropout(vib_to_temp_attn)
        temp_to_vib_attn = self.dropout(temp_to_vib_attn)

        # Compute weighted sum of values
        vib_attended = torch.matmul(vib_to_temp_attn, temp_v)
        temp_attended = torch.matmul(temp_to_vib_attn, vib_v)

        # Reshape back
        vib_attended = vib_attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)
        temp_attended = temp_attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)

        # Output projection
        vib_attended = self.output_vib(vib_attended)
        temp_attended = self.output_temp(temp_attended)

        return vib_attended, temp_attended

class FusionStrategy(nn.Module):
    """
    Implements Early, Mid, and Late fusion strategies.
    """

    def __init__(self, vib_feature_dim, temp_feature_dim, fusion_type="concat"):
        """
        Initialize the Fusion Strategy module.

        Args:
            vib_feature_dim (int): Dimension of vibration features
            temp_feature_dim (int): Dimension of temperature features
            fusion_type (str): Type of fusion - "early", "mid", "late", or "concat"
        """
        super(FusionStrategy, self).__init__()
        self.fusion_type = fusion_type
        self.vib_feature_dim = vib_feature_dim
        self.temp_feature_dim = temp_feature_dim

        if fusion_type == "early":
            # Early fusion: concatenate raw inputs before processing
            self.fusion_layer = nn.Linear(vib_feature_dim + temp_feature_dim, vib_feature_dim)
        elif fusion_type == "mid":
            # Mid fusion: process modalities separately, then fuse features
            self.fusion_layer = nn.Linear(vib_feature_dim + temp_feature_dim, vib_feature_dim)
        elif fusion_type == "late":
            # Late fusion: process modalities separately, fuse at decision level
            self.fusion_layer = nn.Linear(vib_feature_dim + temp_feature_dim, 1)  # For regression
        else:  # concat
            # Simple concatenation
            self.fusion_layer = None

    def forward(self, vib_features, temp_features):
        """
        Forward pass through fusion strategy.

        Args:
            vib_features (torch.Tensor): Vibration features
            temp_features (torch.Tensor): Temperature features

        Returns:
            torch.Tensor: Fused features
        """
        if self.fusion_type == "early":
            # Concatenate and apply fusion layer
            combined = torch.cat([vib_features, temp_features], dim=-1)
            return self.fusion_layer(combined)
        elif self.fusion_type == "mid":
            # Process separately (would be done before this module), then fuse
            combined = torch.cat([vib_features, temp_features], dim=-1)
            return self.fusion_layer(combined)
        elif self.fusion_type == "late":
            # For late fusion, we return both for separate processing
            # The actual fusion happens at the regression level
            return vib_features, temp_features
        else:  # concat
            return torch.cat([vib_features, temp_features], dim=-1)

if __name__ == "__main__":
    # Example usage
    print("Cross-modal Attention module ready")