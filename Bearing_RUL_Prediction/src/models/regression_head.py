"""
Regression head for RUL prediction.
"""
import torch
import torch.nn as nn
from .common import BaseModel
from .config import ModelConfig


class RegressionHead(BaseModel):
    """
    Regression head that maps temporal embedding to RUL prediction.

    Structure:
        -> Linear
        -> activation
        -> dropout
        -> Linear
        -> Softplus (if non_negative=True) or Identity (if non_negative=False)
        -> RUL [B, 1]
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.input_size = config.d_model  # We expect input to be d_model
        self.hidden_sizes = config.regression_hidden
        self.dropout_rate = config.regression_dropout
        self.non_negative = config.non_negative

        # Build the hidden layers
        layers = []
        in_size = self.input_size
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())  # We can make this configurable, but ReLU is common
            layers.append(nn.Dropout(self.dropout_rate))
            in_size = hidden_size
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(in_size, 1)

        # Output activation
        if self.non_negative:
            self.output_activation = nn.Softplus()
        else:
            self.output_activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_model] (we expect the temporal embedding to be pooled or last step)

        Returns:
            RUL prediction: [B, 1]
        """
        # Pass through hidden layers
        x = self.hidden_layers(x)  # [B, last_hidden_size]
        # Output layer
        x = self.output_layer(x)  # [B, 1]
        # Output activation
        x = self.output_activation(x)
        return x