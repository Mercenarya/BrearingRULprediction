"""
Dual-Path LSTM for temporal modeling: fast path for short-term dependencies,
slow path for long-term trends.
"""
import torch
import torch.nn as nn

class DualPathLSTM(nn.Module):
    """
    Dual-Path LSTM with fast and slow pathways.
    """

    def __init__(self, input_size, fast_hidden_size=64, slow_hidden_size=64,
                 fast_num_layers=2, slow_num_layers=2, dropout=0.3):
        """
        Initialize the Dual-Path LSTM.

        Args:
            input_size (int): Number of input features
            fast_hidden_size (int): Hidden size for fast path LSTM
            slow_hidden_size (int): Hidden size for slow path LSTM
            fast_num_layers (int): Number of layers for fast path
            slow_num_layers (int): Number of layers for slow path
            dropout (float): Dropout rate
        """
        super(DualPathLSTM, self).__init__()
        self.input_size = input_size
        self.fast_hidden_size = fast_hidden_size
        self.slow_hidden_size = slow_hidden_size

        # Fast path LSTM - captures short-term dynamics
        self.fast_lstm = nn.LSTM(
            input_size,
            fast_hidden_size,
            num_layers=fast_num_layers,
            batch_first=True,
            dropout=dropout if fast_num_layers > 1 else 0
        )

        # Slow path LSTM - captures long-term trends
        self.slow_lstm = nn.LSTM(
            input_size,
            slow_hidden_size,
            num_layers=slow_num_layers,
            batch_first=True,
            dropout=dropout if slow_num_layers > 1 else 0
        )

        # Adaptive combining layer
        self.combination_layer = nn.Linear(fast_hidden_size + slow_hidden_size, input_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, fast_hidden=None, slow_hidden=None):
        """
        Forward pass through dual-path LSTM.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            fast_hidden (tuple, optional): Initial hidden state for fast path
            slow_hidden (tuple, optional): Initial hidden state for slow path

        Returns:
            tuple: (output, fast_hidden, slow_hidden) where output has shape
                  (batch_size, seq_len, input_size)
        """
        # Fast path processing
        fast_output, fast_hidden = self.fast_lstm(x, fast_hidden)

        # Slow path processing
        slow_output, slow_hidden = self.slow_lstm(x, slow_hidden)

        # Combine fast and slow path outputs
        combined = torch.cat([fast_output, slow_output], dim=-1)
        combined = self.dropout(combined)
        output = self.combination_layer(combined)

        return output, fast_hidden, slow_hidden

    def init_hidden(self, batch_size, device):
        """
        Initialize hidden states for both LSTM paths.

        Args:
            batch_size (int): Batch size
            device (torch.device): Device to place tensors on

        Returns:
            tuple: (fast_hidden, slow_hidden) where each is (h_0, c_0)
        """
        fast_h0 = torch.zeros(self.fast_lstm.num_layers, batch_size, self.fast_hidden_size).to(device)
        fast_c0 = torch.zeros(self.fast_lstm.num_layers, batch_size, self.fast_hidden_size).to(device)
        slow_h0 = torch.zeros(self.slow_lstm.num_layers, batch_size, self.slow_hidden_size).to(device)
        slow_c0 = torch.zeros(self.slow_lstm.num_layers, batch_size, self.slow_hidden_size).to(device)

        return (fast_h0, fast_c0), (slow_h0, slow_c0)

if __name__ == "__main__":
    # Example usage
    print("Dual-Path LSTM module ready")