"""
Models module for Bearing RUL Prediction.
"""

from .config import ModelConfig
from .factory import build_model, list_available_models, register_model
from .common import BaseModel

# Encoders
from .vibration_encoder import VibrationEncoder
from .temperature_encoder import TemperatureEncoder

# Attention and fusion
from .cross_attention import CrossModalAttention
from .scale_fusion import ScaleFusion

# Temporal
from .temporal import TemporalEncoder, DualPathTemporalEncoder

# Regression head
from .regression_head import RegressionHead

# Baselines
from .baselines import (
    VibrationCNNRegressor,
    TemperatureRegressor,
    LateConcatFusionRegressor,
    CNNTemporalRegressor
)

# Full model
from .multimodal_rul import MultimodalRUL

__all__ = [
    "ModelConfig",
    "BaseModel",
    "VibrationEncoder",
    "TemperatureEncoder",
    "CrossModalAttention",
    "ScaleFusion",
    "TemporalEncoder",
    "DualPathTemporalEncoder",
    "RegressionHead",
    "VibrationCNNRegressor",
    "TemperatureRegressor",
    "LateConcatFusionRegressor",
    "CNNTemporalRegressor",
    "MultimodalRUL",
    "build_model",
    "list_available_models",
    "register_model",
]