"""
Regression Head: Fully connected network for RUL prediction.
Architecture: FC 192→128→64→1
"""
import torch
import torch.nn as nn

class RegressionHead(nn.Module):
    """
    Regression head for predicting RUL from extracted features.
    """

    def __init__(self, input_size=192, hidden_sizes=[128, 64], output_size=1,
                 dropout_rate=0.3):
        """
        Initialize the Regression Head.

        Args:
            input_size (int): Size of input features
            hidden_sizes (list): Sizes of hidden layers [128, 64]
            output_size (int): Size of output (1 for RUL regression)
            dropout_rate (float): Dropout rate between layers
        """
        super(RegressionHead, self).__init__()

        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the regression head.

        Args:
            x (torch.Tensor): Input features of shape (batch_size, input_size)
                             or (batch_size, seq_len, input_size) for temporal data

        Returns:
            torch.Tensor: Predicted RUL values of shape (batch_size, output_size)
                         or (batch_size, seq_len, output_size) for temporal data
        """
        # Handle temporal data by applying to each time step
        original_shape = x.shape
        if len(original_shape) == 3:
            batch_size, seq_len, feature_size = original_shape
            x = x.view(-1, feature_size)  # (batch_size * seq_len, input_size)
            output = self.network(x)
            output = output.view(batch_size, seq_len, -1)  # (batch_size, seq_len, output_size)
        else:
            output = self.network(x)

        return output

    def predict_rul(self, features):
        """
        Predict RUL from features (convenience method).

        Args:
            features (torch.Tensor): Input features

        Returns:
            torch.Tensor: Predicted RUL values
        """
        return self.forward(features)

if __name__ == "__main__":
    # Example usage
    print("Regression Head module ready")