"""
Ablation study script for Bearing RUL prediction model.
Runs 3-step ablation:
1. Remove cross-modal attention
2. Replace dual-path LSTM with single LSTM
3. Replace stage-aware loss with standard MSE
"""
import os
import copy
import yaml
import torch
import numpy as np
from src.data.dataset import BearingDataset, create_data_loaders
from src.models.rul_model import RULModel
from src.models.baselines.cnn_bilstm import CNNBiLSTM
from src.training.trainer import Trainer
from src.losses.stage_aware_loss import StageAwarePHMLoss
import torch.nn as nn

def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_ablation_model(base_config, ablation_type):
    """
    Create model for specific ablation study.

    Args:
        base_config (dict): Base configuration
        ablation_type (str): Type of ablation ('no_attention', 'single_lstm', 'mse_loss')

    Returns:
        nn.Module: Model for ablation study
    """
    config = copy.deepcopy(base_config)

    if ablation_type == 'no_attention':
        # Remove cross-modal attention by using simple concatenation fusion
        config['model']['fusion_type'] = 'concat'
        # We'll use a modified RULModel without attention - for simplicity,
        # we'll use the CNNBiLSTM baseline which doesn't have attention
        model = CNNBiLSTM(
            vib_in_channels=config['data'].get('vib_channels', 1),
            temp_in_channels=config['data'].get('temp_channels', 1),
            cnn_out_channels=config['model'].get('cnn_out_channels', 64),
            lstm_hidden_size=config['model'].get('lstm_hidden_size', 128),
            num_lstm_layers=config['model'].get('lstm_num_layers', 2),
            regression_hidden=config['model'].get('regression_hidden', [128, 64]),
            dropout_rate=config['model'].get('dropout_rate', 0.3)
        )
        return model

    elif ablation_type == 'single_lstm':
        # Replace dual-path LSTM with single LSTM
        # For this ablation, we'll create a custom model
        from src.models.cnn_encoder import VibrationEncoder, TemperatureEncoder
        from src.models.cross_modal_attention import CrossModalAttention, FusionStrategy
        from src.models.regression_head import RegressionHead
        import torch.nn as nn

        class SingleLSTMModel(nn.Module):
            def __init__(self, vib_in_channels, temp_in_channels,
                         cnn_out_channels=64, lstm_hidden_size=128,
                         fusion_type="concat", num_lstm_layers=2,
                         regression_hidden=[128, 64], dropout_rate=0.3):
                super(SingleLSTMModel, self).__init__()

                # CNN Encoders
                self.vib_encoder = VibrationEncoder(vib_in_channels, cnn_out_channels)
                self.temp_encoder = TemperatureEncoder(temp_in_channels, cnn_out_channels)

                # Feature dimensions
                self.vib_feature_dim = 3 * cnn_out_channels
                self.temp_feature_dim = 3 * cnn_out_channels

                # Cross-modal attention
                self.cross_modal_attn = CrossModalAttention(
                    feature_dim=self.vib_feature_dim,
                    num_heads=4
                )

                # Fusion
                self.fusion = FusionStrategy(
                    vib_feature_dim=self.vib_feature_dim,
                    temp_feature_dim=self.temp_feature_dim,
                    fusion_type=fusion_type
                )

                # Determine LSTM input size
                if fusion_type == "late":
                    self.lstm_input_size = self.vib_feature_dim
                else:
                    if fusion_type == "concat":
                        self.lstm_input_size = self.vib_feature_dim + self.temp_feature_dim
                    else:
                        self.lstm_input_size = self.vib_feature_dim

                # Single LSTM (replacing dual-path)
                self.lstm = nn.LSTM(
                    input_size=self.lstm_input_size,
                    hidden_size=lstm_hidden_size,
                    num_layers=num_lstm_layers,
                    batch_first=True,
                    dropout=dropout_rate if num_lstm_layers > 1 else 0
                )

                # Regression Head
                self.regression_head = RegressionHead(
                    input_size=lstm_hidden_size,
                    hidden_sizes=regression_hidden,
                    output_size=1,
                    dropout_rate=dropout_rate
                )

                self.dropout = nn.Dropout(dropout_rate)

            def forward(self, vib_input, temp_input):
                batch_size, seq_len, _ = vib_input.shape

                # CNN Encoding
                vib_transposed = vib_input.transpose(1, 2)
                temp_transposed = temp_input.transpose(1, 2)

                vib_features = self.vib_encoder(vib_transposed)
                temp_features = self.temp_encoder(temp_transposed)

                vib_features = vib_features.transpose(1, 2)
                temp_features = temp_features.transpose(1, 2)

                # Cross-modal attention
                vib_attended, temp_attended = self.cross_modal_attn(vib_features, temp_features)

                # Fusion
                fused_features = self.fusion(vib_attended, temp_attended)
                fused_features = self.dropout(fused_features)

                # Single LSTM
                lstm_output, _ = self.lstm(fused_features)

                # Take last time step
                lstm_features = lstm_output[:, -1, :]

                # Regression head
                rul_prediction = self.regression_head(lstm_features)

                return rul_prediction

        model = SingleLSTMModel(
            vib_in_channels=config['data'].get('vib_channels', 1),
            temp_in_channels=config['data'].get('temp_channels', 1),
            cnn_out_channels=config['model'].get('cnn_out_channels', 64),
            lstm_hidden_size=config['model'].get('lstm_hidden_size', 128),
            fusion_type=config['model'].get('fusion_type', 'concat'),
            num_lstm_layers=config['model'].get('lstm_num_layers', 2),
            regression_hidden=config['model'].get('regression_hidden', [128, 64]),
            dropout_rate=config['model'].get('dropout_rate', 0.3)
        )
        return model

    elif ablation_type == 'mse_loss':
        # Keep full model but change loss function to MSE
        model = RULModel(
            vib_in_channels=config['data'].get('vib_channels', 1),
            temp_in_channels=config['data'].get('temp_channels', 1),
            cnn_out_channels=config['model'].get('cnn_out_channels', 64),
            lstm_hidden_size=config['model'].get('lstm_hidden_size', 128),
            fusion_type=config['model'].get('fusion_type', 'concat'),
            num_lstm_layers=config['model'].get('lstm_num_layers', 2),
            regression_hidden=config['model'].get('regression_hidden', [128, 64]),
            dropout_rate=config['model'].get('dropout_rate', 0.3)
        )
        return model

    else:
        # Full model (no ablation)
        model = RULModel(
            vib_in_channels=config['data'].get('vib_channels', 1),
            temp_in_channels=config['data'].get('temp_channels', 1),
            cnn_out_channels=config['model'].get('cnn_out_channels', 64),
            lstm_hidden_size=config['model'].get('lstm_hidden_size', 128),
            fusion_type=config['model'].get('fusion_type', 'concat'),
            num_lstm_layers=config['model'].get('lstm_num_layers', 2),
            regression_hidden=config['model'].get('regression_hidden', [128, 64]),
            dropout_rate=config['model'].get('dropout_rate', 0.3)
        )
        return model

def run_ablation_study(ablation_types=None):
    """
    Run ablation study comparing different model configurations.

    Args:
        ablation_types (list): List of ablation types to test.
                              If None, tests all standard ablations.
    """
    if ablation_types is None:
        ablation_types = ['full_model', 'no_attention', 'single_lstm', 'mse_loss']

    # Load base configuration
    config = load_config()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_path=os.path.join('data', 'processed', 'train.npz'),
        val_path=os.path.join('data', 'processed', 'val.npz'),
        test_path=os.path.join('data', 'processed', 'test.npz'),
        batch_size=config['training']['batch_size'],
        num_workers=0
    )

    # Results storage
    results = {}

    for ablation_type in ablation_types:
        print(f"\n{'='*60}")
        print(f"Running ablation study: {ablation_type}")
        print(f"{'='*60}")

        # Create model for this ablation
        model = create_ablation_model(config, ablation_type)
        model = model.to(device)

        # Setup loss function
        if ablation_type == 'mse_loss':
            # Use MSE loss for this ablation
            criterion = torch.nn.MSELoss()
        else:
            # Use stage-aware PHM loss
            loss_config = config.get('loss', {})
            criterion = StageAwarePHMLoss(
                alpha=loss_config.get('alpha', 0.5),
                beta=loss_config.get('beta', 0.3),
                theta=loss_config.get('theta', 0.1)
            )

        criterion = criterion.to(device) if hasattr(criterion, 'to') else criterion

        # Setup optimizer
        optimizer_name = config['training'].get('optimizer', 'adam')
        learning_rate = config['training']['learning_rate']

        if optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Setup trainer
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            early_stopping_patience=config['training'].get('early_stopping_patience', 10),
            checkpoint_dir=os.path.join('checkpoints', f'ablation_{ablation_type}'),
            use_wandb=False  # Disable wandb for ablation studies to avoid clutter
        )

        # Train model
        print(f"Training {ablation_type} model...")
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs']
        )

        # Evaluate on test set
        print(f"Evaluating {ablation_type} model on test set...")
        test_loss, test_metrics = trainer.evaluate(test_loader)

        # Store results
        results[ablation_type] = {
            'test_loss': test_loss,
            'test_metrics': test_metrics
        }

        print(f"Results for {ablation_type}:")
        for metric_name, metric_value in test_metrics.items():
            print(f"  {metric_name.upper()}: {metric_value:.4f}")

    # Print summary
    print(f"\n{'='*60}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*60}")

    # Sort by PHM score (lower is better)
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]['test_metrics']['phm_score']
    )

    print(f"{'Ablation Type':<20} {'PHM Score':<12} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
    print("-" * 70)
    for ablation_type, result in sorted_results:
        metrics = result['test_metrics']
        print(f"{ablation_type:<20} {metrics['phm_score']:<12.4f} {metrics['mae']:<12.4f} "
              f"{metrics['rmse']:<12.4f} {metrics['r2']:<12.4f}")

    # Save results
    results_dir = os.path.join('results', 'tables')
    os.makedirs(results_dir, exist_ok=True)

    import json
    with open(os.path.join(results_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to {results_dir}/ablation_results.json")

if __name__ == "__main__":
    print("Ablation Study for Bearing RUL Prediction Model")
    print("This script will run 3-step ablation:")
    print("1. Full model (baseline)")
    print("2. Remove cross-modal attention")
    print("3. Replace dual-path LSTM with single LSTM")
    print("4. Replace stage-aware loss with standard MSE")
    print()

    run_ablation_study()