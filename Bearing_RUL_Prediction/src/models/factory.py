"""
Model factory for building models by name.
"""
from typing import Dict, Type, Optional, Any, List
import torch.nn as nn
from .config import ModelConfig

# Import model classes (will be defined later)
from .multimodal_rul import MultimodalRUL
from .baselines import (
    VibrationCNNRegressor,
    TemperatureRegressor,
    LateConcatFusionRegressor,
    CNNTemporalRegressor,
    SingleScaleCrossAttentionRegressor,
    MultiScaleFusionRegressor,
    MultiScaleCrossAttentionRegressor
)

# Registry of model names to classes
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    # Baselines
    "vib_cnn": VibrationCNNRegressor,
    "temp_mlp": TemperatureRegressor,
    "late_concat": LateConcatFusionRegressor,
    "cnn_gru": CNNTemporalRegressor,
    # Ablation studies
    "single_scale_cross_attention": SingleScaleCrossAttentionRegressor,
    "multi_scale_fusion": MultiScaleFusionRegressor,
    "multi_scale_cross_attention": MultiScaleCrossAttentionRegressor,
    # Full model
    "full_dual_path": MultimodalRUL,
}


def build_model(model_name: str, config: ModelConfig) -> nn.Module:
    """
    Build a model by name.

    Args:
        model_name: Key in MODEL_REGISTRY.
        config: ModelConfig instance.

    Returns:
        An instance of nn.Module.

    Raises:
        ValueError: If model_name is not in registry.
        NotImplementedError: If the model class is not yet implemented.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]
    if model_class is None:
        raise NotImplementedError(f"Model '{model_name}' is not yet implemented.")

    # Prepare config for the model (some models might need only a subset)
    # We pass the whole config and let the model extract what it needs.
    return model_class(config)


def register_model(name: str, model_class: Type[nn.Module]) -> None:
    """
    Register a new model class.

    Args:
        name: Model key.
        model_class: Class of the model (must inherit from nn.Module).
    """
    MODEL_REGISTRY[name] = model_class


def list_available_models() -> List[str]:
    """Return a list of available model names."""
    return list(MODEL_REGISTRY.keys())