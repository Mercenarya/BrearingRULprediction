"""
Unit tests for the stage-aware PHM loss function.
"""
import unittest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from losses.stage_aware_loss import StageAwarePHMLoss, WeightedMAELoss


class TestStageAwarePHMLoss(unittest.TestCase):
    """Test cases for StageAwarePHMLoss class."""

    def setUp(self):
        """Set up test fixtures."""
        self.loss_fn = StageAwarePHMLoss(alpha=0.5, beta=0.3, theta=0.1)
        self.weighted_loss_fn = WeightedMAELoss(early_weight=2.0, late_weight=1.5, theta=0.1)

    def test_basic_functionality(self):
        """Test basic loss computation."""
        # Simple test case
        y_pred = torch.tensor([[1.0], [2.0], [3.0]])
        y_true = torch.tensor([[1.0], [2.0], [3.0]])

        loss = self.loss_fn(y_pred, y_true)

        # When predictions are perfect, loss should be zero (or very close)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_perfect_prediction(self):
        """Test loss when predictions are exactly correct."""
        y_pred = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        y_true = torch.tensor([[0.0], [1.0], [2.0], [3.0]])

        loss = self.loss_fn(y_pred, y_true)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_under_prediction_penalty(self):
        """Test that under-prediction is penalized according to alpha parameter."""
        # Case where we under-predict (y_pred < y_true)
        y_pred = torch.tensor([[1.0]])
        y_true = torch.tensor([[2.0]])  # Under-predicted by 1.0

        loss = self.loss_fn(y_pred, y_true)

        # Manual calculation:
        # error = y_true - y_pred = 1.0
        # Since y_true (2.0) > theta (0.1), this is early stage
        # loss = exp(alpha * error) - 1 = exp(0.5 * 1.0) - 1 = exp(0.5) - 1
        expected_loss = torch.exp(0.5 * 1.0) - 1

        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=6)


    def test_over_prediction_penalty(self):
        """Test that over-prediction is penalized according to beta parameter."""
        # Case where we over-predict (y_pred > y_true)
        y_pred = torch.tensor([[3.0]])
        y_true = torch.tensor([[2.0]])  # Over-predicted by 1.0

        loss = self.loss_fn(y_pred, y_true)

        # Manual calculation:
        # error = y_true - y_pred = -1.0
        # Since y_true (2.0) > theta (0.1), this is early stage
        # loss = exp(-beta * error) - 1 = exp(-0.3 * (-1.0)) - 1 = exp(0.3) - 1
        expected_loss = torch.exp(0.3 * 1.0) - 1
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=6)




    def test_stage_boundary_behavior(self):
        """Test behavior at the stage boundary (theta)."""
        theta = 0.1

        # Test early stage (just above theta)
        y_pred_early = torch.tensor([[0.0]])
        y_true_early = torch.tensor([[theta + 0.01]])  # 0.11

        loss_early = self.loss_fn(y_pred_early, y_true_early)

        # Test late stage (at theta)
        y_pred_late = torch.tensor([[0.0]])
        y_true_late = torch.tensor([[theta]])  # 0.10

        loss_late = self.loss_fn(y_pred_late, y_true_late)

        # Test late stage (just below theta)
        y_pred_late2 = torch.tensor([[0.0]])
        y_true_late2 = torch.tensor([[theta - 0.01]])  # 0.09

        loss_late2 = self.loss_fn(y_pred_late2, y_true_late2)

        # Early and late stage losses should be different due to different weighting
        # Note: For these specific values (small error), the difference might be subtle
        # but they should not be identical due to different formulas applied
        self.assertNotEqual(low_loss.item(), loss_late2.item())

    def test_batch_processing(self):
        """Test loss computation with batches."""
        batch_size = 10
        y_pred = torch.randn(batch_size, 1)
        y_true = torch.randn(batch_size, 1)

        loss = self.loss_fn(y_pred, y_true)

        # Loss should be a scalar
        self.assertEqual(loss.dim(), 0)  # Scalar tensor
        self.assertIsInstance(loss.item(), float)

    def test_reduction_methods(self):
        """Test different reduction methods."""
        y_pred = torch.tensor([[1.0], [2.0], [3.0]])
        y_true = torch.tensor([[1.5], [2.5], [3.5]])

        # Test mean reduction
        loss_mean = StageAwarePHMLoss(reduction='mean')
        mean_loss = loss_mean(y_pred, y_true)

        # Test sum reduction
        loss_sum = StageAwarePHMLoss(reduction='sum')
        sum_loss = loss_sum(y_pred, y_true)

        # Test none reduction
        loss_none = StageAwarePHMLoss(reduction='none')
        none_loss = loss_none(y_pred, y_true)

        # Check shapes
        self.assertEqual(mean_loss.dim(), 0)  # Scalar
        self.assertEqual(sum_loss.dim(), 0)   # Scalar
        self.assertEqual(none_loss.shape, torch.Size([3, 1]))  # Same as input

        # Check relationships
        self.assertAlmostEqual(sum_loss.item(), 3 * mean_loss.item(), places=6)
        self.assertAlmostEqual(torch.mean(none_loss).item(), mean_loss.item(), places=6)

    def test_edge_cases(self):
        """Test edge cases."""
        # Zero error
        y_pred = torch.tensor([[1.0]])
        y_true = torch.tensor([[1.0]])
        loss = self.loss_fn(y_pred, y_true)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

        # Large values
        y_pred = torch.tensor([[100.0]])
        y_true = torch.tensor([[50.0]])
        loss = self.loss_fn(y_pred, y_true)
        # Should not produce NaN or infinite values
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

        # Negative values
        y_pred = torch.tensor([[-10.0]])
        y_true = torch.tensor([[5.0]])
        loss = self.loss_fn(y_pred, y_true)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))


    def test_different_parameters(self):
        """Test loss with different alpha, beta, theta values."""
        y_pred = torch.tensor([[0.0]])
        y_true = torch.tensor([[1.0]])  # Under-prediction of 1.0

        # Test different alpha values
        loss_alpha_low = StageAwarePHMLoss(alpha=0.1, beta=0.3, theta=0.1)(y_pred, y_true)
        loss_alpha_high = StageAwarePHMLoss(alpha=1.0, beta=0.3, theta=0.1)(y_pred, y_true)

        # Higher alpha should give higher loss for under-prediction
        self.assertGreater(loss_alpha_high.item(), loss_alpha_low.item())

        # Test different beta values
        y_pred_over = torch.tensor([[2.0]])  # Over-prediction of 1.0 (y_true=1.0)
        loss_beta_low = StageAwarePHMLoss(alpha=0.5, beta=0.1, theta=0.1)(y_pred_over, y_true)
        loss_beta_high = StageAwarePHMLoss(alpha=0.5, beta=1.0, theta=0.1)(y_pred_over, y_true)

        # Higher beta should give higher loss for over-prediction
        self.assertGreater(loss_beta_high.item(), loss_beta_low.item())

        # Test different theta values
        y_pred = torch.tensor([[0.0]])
        y_true_high = torch.tensor([[0.5]])  # Well above typical theta
        y_true_low = torch.tensor([[0.01]])  # Well below typical theta

        loss_theta_low = StageAwarePHMLoss(alpha=0.5, beta=0.3, theta=0.01)(y_pred, y_true_high)
        loss_theta_high = StageAwarePHMLoss(alpha=0.5, beta=0.3, theta=0.5)(y_pred, y_true_low)

        # With low theta, 0.5 is in late stage (over-prediction penalty)
        # With high theta, 0.5 is in early stage (under-prediction penalty)
        # These should produce different loss values
        self.assertNotEqual(loss_theta_low.item(), loss_theta_high.item())

    def test_weighted_mae_loss(self):
        """Test the WeightedMAELoss alternative implementation."""
        y_pred = torch.tensor([[1.0], [2.0]])
        y_true = torch.tensor([[2.0], [1.0]])  # Both have error of 1.0

        # First sample: y_true=2.0 (> theta=0.1) -> early stage -> weight=2.0
        # Second sample: y_true=1.0 (> theta=0.1) -> early stage -> weight=2.0
        # Both are early stage in this case

        loss = self.weighted_loss_fn(y_pred, y_true)

        # Manual calculation:
        # Errors: [1.0, 1.0] (absolute)
        # Weights: [2.0, 2.0] (both early stage)
        # Weighted errors: [2.0, 2.0]
        # Mean: 2.0

        expected_loss = 2.0
        self.assertAlmostEqual(loss.item(), expected_loss, places=6)

        # Test with mixed stages
        y_pred = torch.tensor([[0.0], [0.0]])
        y_true = torch.tensor([[0.2], [0.05]])  # 0.2 > theta (early), 0.05 < theta (late)

        loss = self.weighted_loss_fn(y_pred, y_true)

        # First sample: error=0.2, early stage -> weight=2.0 -> weighted error=0.4
        # Second sample: error=0.05, late stage -> weight=1.5 -> weighted error=0.075
        # Mean weighted error: (0.4 + 0.075) / 2 = 0.2375

        expected_loss = 0.2375
        self.assertAlmostEqual(loss.item(), expected_loss, places=6)

if __name__ == '__main__':
    unittest.main()