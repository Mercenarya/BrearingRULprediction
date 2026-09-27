"""
Stage-aware loss function: PHM Loss asymmetric according to stage.
Implements the Prognostics and Health Management (PHM) loss with
asymmetric weighting based on RUL stages.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class StageAwarePHMLoss(nn.Module):
    """
    Stage-aware PHM Loss with asymmetric weighting.
    """

    def __init__(self, alpha=0.5, beta=0.3, theta=0.1, reduction='mean'):
        """
        Initialize the Stage-aware PHM Loss.

        Args:
            alpha (float): Weight for early stage (high RUL) under-prediction penalty
            beta (float): Weight for late stage (low RUL) over-prediction penalty
            theta (float): Threshold parameter defining early/late stage boundary
            reduction (str): Specifies the reduction to apply to the output:
                           'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super(StageAwarePHMLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        Forward pass for stage-aware PHM loss.

        Args:
            y_pred (torch.Tensor): Predicted RUL values
            y_true (torch.Tensor): True RUL values

        Returns:
            torch.Tensor: Computed loss value
        """
        # Calculate error
        error = y_true - y_pred  # Positive error means under-prediction

        # Define stages based on true RUL values
        # Early stage: high RUL (y_true > theta)
        # Late stage: low RUL (y_true <= theta)
        early_stage = (y_true > self.theta).float()
        late_stage = (y_true <= self.theta).float()

        # Asymmetric loss components
        # Under-prediction penalty (when we predict lower RUL than actual)
        under_penalty = torch.exp(self.alpha * error) - 1
        # Over-prediction penalty (when we predict higher RUL than actual)
        over_penalty = torch.exp(-self.beta * error) - 1

        # Apply stage-specific weighting
        # Early stage: penalize under-prediction more heavily
        # Late stage: penalize over-prediction more heavily
        loss = early_stage * under_penalty + late_stage * over_penalty

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:  # 'none'
            return loss

class WeightedMAELoss(nn.Module):
    """
    Weighted MAE loss with stage-dependent weights.
    Alternative implementation of stage-aware loss.
    """

    def __init__(self, early_weight=2.0, late_weight=1.5, theta=0.1):
        """
        Initialize Weighted MAE Loss.

        Args:
            early_weight (float): Weight for errors in early stage
            late_weight (float): Weight for errors in late stage
            theta (float): Threshold for stage separation
        """
        super(WeightedMAELoss, self).__init__()
        self.early_weight = early_weight
        self.late_weight = late_weight
        self.theta = theta

    def forward(self, y_pred, y_true):
        """
        Forward pass for weighted MAE loss.

        Args:
            y_pred (torch.Tensor): Predicted RUL values
            y_true (torch.Tensor): True RUL values

        Returns:
            torch.Tensor: Computed loss value
        """
        # Calculate absolute error
        abs_error = torch.abs(y_true - y_pred)

        # Define stages
        early_stage = (y_true > self.theta).float()
        late_stage = (y_true <= self.theta).float()

        # Apply stage-dependent weights
        weighted_error = early_weight * early_stage * abs_error + \
                        late_weight * late_stage * abs_error

        return torch.mean(weighted_error)

if __name__ == "__main__":
    # Example usage
    print("Stage-aware PHM Loss module ready")