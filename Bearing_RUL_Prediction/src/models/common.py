"""
Common utilities and base classes for models.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from .config import ModelConfig


class BaseModel(nn.Module):
    """
    Base class for all models.
    Defines the interface and common methods.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            batch: Dictionary containing:
                - vibration: [B, T, 2, L_vib]
                - temperature: [B, T, F_temp] or [B, T, 1, 64]
                - temperature_mask: [B, T, F_temp] or [B, T, 1, 64] (bool)
                - rul: [B, 1] (not used in forward, but present in batch)

        Returns:
            Dictionary with:
                - rul_pred: [B, 1]
                - aux: dict (optional, if config.return_aux=True)
        """
        raise NotImplementedError

    def predict_rul(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict RUL in evaluation mode.

        Args:
            batch: Same as forward.

        Returns:
            RUL prediction tensor [B, 1].
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(batch)
            return output["rul_pred"]

    def get_output_shape(self, batch: Dict[str, torch.Tensor]) -> Tuple[int, ...]:
        """
        Get the output shape without running the model (for testing).
        """
        # This is a placeholder; subclasses should implement if needed.
        raise NotImplementedError

    def parameter_count(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def validate_batch(batch: Dict[str, torch.Tensor]) -> None:
    """
    Validate the batch tensor shapes and types.
    """
    required_keys = ["vibration", "temperature", "temperature_mask", "rul"]
    for key in required_keys:
        if key not in batch:
            raise KeyError(f"Missing key '{key}' in batch")

    # Check types
    if not isinstance(batch["vibration"], torch.Tensor):
        raise TypeError("vibration must be a torch.Tensor")
    if not isinstance(batch["temperature"], torch.Tensor):
        raise TypeError("temperature must be a torch.Tensor")
    if not isinstance(batch["temperature_mask"], torch.Tensor):
        raise TypeError("temperature_mask must be a torch.Tensor")
    if not isinstance(batch["rul"], torch.Tensor):
        raise TypeError("rul must be a torch.Tensor")

    # Check shapes (we can only check the known dimensions)
    B = batch["vibration"].shape[0]
    T = batch["vibration"].shape[1]
    if batch["temperature"].shape[0] != B or batch["temperature"].shape[1] != T:
        raise ValueError("Batch and sequence length must match across modalities")
    if batch["temperature_mask"].shape[0] != B or batch["temperature_mask"].shape[1] != T:
        raise ValueError("Batch and sequence length must match for temperature_mask")
    if batch["rul"].shape[0] != B or batch["rul"].shape[1] != 1:
        raise ValueError("rul must have shape [B, 1]")


def get_activation(activation: str) -> nn.Module:
    """
    Get activation function by name.
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Unsupported activation: {activation}")


def get_normalization(norm_type: str, num_features: int) -> nn.Module:
    """
    Get normalization layer by name.
    """
    if norm_type == "batch":
        return nn.BatchNorm1d(num_features)
    elif norm_type == "layer":
        return nn.LayerNorm(num_features)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")