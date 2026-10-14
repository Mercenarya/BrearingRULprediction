"""
Temporal encoder modules.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from .common import BaseModel
from .config import ModelConfig


class TemporalEncoder(BaseModel):
    """
    Simple temporal encoder using GRU or LSTM.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.d_model = config.d_model
        self.temporal_type = config.temporal_type
        self.hidden_size = config.temporal_hidden_size
        self.num_layers = config.temporal_num_layers
        self.dropout = config.temporal_dropout
        self.bidirectional = config.bidirectional
        self.return_aux = config.return_aux

        if self.temporal_type == "gru":
            self.rnn = nn.GRU(
                input_size=self.d_model,  # We expect input to be d_model
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                batch_first=True,
            )
        elif self.temporal_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=self.d_model,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported temporal type: {self.temporal_type}")

        # The output dimension of the RNN
        self.output_dim = self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            Tuple of (output, hidden, cell, temporal_aux) where:
                - output: [B, T, output_dim]
                - hidden: [num_layers * num_directions, B, hidden_size] (for LSTM, hidden and cell)
                - cell: same as hidden for LSTM, None for GRU
                - temporal_aux: dict containing temporal weights if return_aux=True, else empty dict
        """
        temporal_aux = {}
        if self.temporal_type == "gru":
            output, hidden = self.rnn(x)
            return output, hidden, None, temporal_aux
        else:  # lstm
            output, (hidden, cell) = self.rnn(x)
            return output, hidden, cell, temporal_aux


class DualPathTemporalEncoder(BaseModel):
    """
    Dual-path temporal encoder with fast and slow paths.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.input_size = config.d_model  # We expect input to be d_model
        self.fast_hidden_size = config.temporal_hidden_size
        self.slow_hidden_size = config.temporal_hidden_size
        self.fast_num_layers = config.temporal_num_layers
        self.slow_num_layers = config.temporal_num_layers
        self.dropout = config.temporal_dropout
        self.return_aux = config.return_aux

        # Fast path: GRU or LSTM
        self.fast_rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.fast_hidden_size,
            num_layers=self.fast_num_layers,
            dropout=self.dropout if self.fast_num_layers > 1 else 0,
            batch_first=True,
        )
        # Slow path: GRU or LSTM
        self.slow_rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.slow_hidden_size,
            num_layers=self.slow_num_layers,
            dropout=self.dropout if self.slow_num_layers > 1 else 0,
            batch_first=True,
        )

        # Projection layers to map hidden states to a common dimension
        # We'll project to d_model for fusion
        self.fast_proj = nn.Linear(self.fast_hidden_size, config.d_model)
        self.slow_proj = nn.Linear(self.slow_hidden_size, config.d_model)

        # For gated fusion, we'll use a gate network
        self.gate_net = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 2),  # Outputs two gates (for fast and slow)
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            Tuple of (output, hidden, cell, temporal_aux) where:
                - output: [B, T, d_model] (fused fast and slow features)
                - hidden: tuple of (fast_hidden, slow_hidden) for compatibility? We'll return None for now.
                - cell: tuple of (fast_cell, slow_cell) or None.
                - temporal_aux: dict containing temporal weights if return_aux=True, else empty dict
        """
        temporal_aux = {}
        # Fast path
        fast_output, (fast_hidden, fast_cell) = self.fast_rnn(x)  # [B, T, fast_hidden_size]
        # Slow path
        slow_output, (slow_hidden, slow_cell) = self.slow_rnn(x)  # [B, T, slow_hidden_size]

        # Project to d_model
        fast_proj = self.fast_proj(fast_output)  # [B, T, d_model]
        slow_proj = self.slow_proj(slow_output)  # [B, T, d_model]

        # Gated fusion: compute gates from the concatenated projected features
        concatenated = torch.cat([fast_proj, slow_proj], dim=-1)  # [B, T, 2*d_model]
        gates = self.gate_net(concatenated)  # [B, T, 2]
        fast_gate = gates[..., 0:1]  # [B, T, 1]
        slow_gate = gates[..., 1:2]  # [B, T, 1]

        # Fused output
        fused = fast_gate * fast_proj + slow_gate * slow_proj  # [B, T, d_model]

        # We return the fused output and the hidden states for compatibility
        # We'll return the hidden states as a tuple of (fast_hidden, slow_hidden) and similarly for cell.
        # But the base TemporalEncoder expects a single hidden and cell? We'll change the return
        # to match the simple TemporalEncoder? Actually, we are not using the base class
        # for dual-path in the same way. We'll return the fused output and then None for hidden and cell
        # to match the simple encoder's return type? But the simple encoder returns (output, hidden, cell).
        # We'll return (fused, None, None) for simplicity, or we can return the hidden states.
        # Let's return the hidden states as a tuple of two tuples? We'll change the return type
        # to be flexible. However, to keep the interface consistent with the simple encoder,
        # we'll return (fused, None, None) and note that the dual-path encoder does not
        # return the hidden states in the same format.

        # Alternatively, we can return a dictionary? We'll stick to the simple encoder's
        # return type for now and return None for hidden and cell.
        if self.return_aux:
            # Return the gates as temporal weights
            temporal_aux["temporal_weights"] = gates  # [B, T, 2]
        return fused, None, None, temporal_aux