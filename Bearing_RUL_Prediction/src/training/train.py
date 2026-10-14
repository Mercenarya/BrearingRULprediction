"""
Main training loop for Bearing RUL prediction models.
"""
import os,sys

CURRENT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(CURRENT,"..",".."))
sys.path.append(ROOT)

import yaml
import torch
import torch.nn as nn
from src.data.dataset import create_data_loaders
from src.models.rul_model import RULModel
from src.losses.stage_aware_loss import StageAwarePHMLoss
from src.training.trainer import Trainer

CONF = os.path.join(ROOT,"config.yaml")
print(ROOT)

def load_config(config_path=CONF):
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_model(config):
    """
    Setup the RUL model based on configuration.

    Args:
        config (dict): Configuration dictionary

    Returns:
        nn.Module: Initialized model
    """
    model_config = config['model']
    data_config = config['data']

    model = RULModel(
        vib_in_channels=data_config.get('vib_channels', 1),
        temp_in_channels=data_config.get('temp_channels', 1),
        cnn_out_channels=model_config.get('cnn_out_channels', 64),
        lstm_hidden_size=model_config.get('lstm_hidden_size', 128),
        fusion_type=model_config.get('fusion_type', 'concat'),
        num_lstm_layers=model_config.get('lstm_num_layers', 2),
        regression_hidden=model_config.get('regression_hidden', [128, 64]),
        dropout_rate=model_config.get('dropout_rate', 0.3)
    )
    return model

def setup_loss_function(config):
    """
    Setup loss function based on configuration.

    Args:
        config (dict): Configuration dictionary

    Returns:
        nn.Module: Loss function
    """
    loss_config = config.get('loss', {})
    loss_type = loss_config.get('type', 'stage_aware_phm')

    if loss_type == 'stage_aware_phm':
        loss_fn = StageAwarePHMLoss(
            alpha=loss_config.get('alpha', 0.5),
            beta=loss_config.get('beta', 0.3),
            theta=loss_config.get('theta', 0.1)
        )
    else:
        # Default to MSE loss
        loss_fn = nn.MSELoss()

    return loss_fn

def main():
    """
    Main training function.
    """
    # Load configuration
    config = load_config()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_path=os.path.join(ROOT,'data', 'processed', 'train.npz'),
        val_path=os.path.join(ROOT,'data', 'processed', 'val.npz'),
        test_path=os.path.join(ROOT,'data', 'processed', 'test.npz'),
        batch_size=config['training']['batch_size'],
        num_workers=config.get('training', {}).get('num_workers', 0)
    )

    # Setup model
    model = setup_model(config)
    model = model.to(device)

    # Setup loss function
    criterion = setup_loss_function(config)
    criterion = criterion.to(device) if hasattr(criterion, 'to') else criterion

    # Setup optimizer
    optimizer_name = config['training'].get('optimizer', 'adam')
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0.0)

    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Setup trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        early_stopping_patience=config['training'].get('early_stopping_patience', 10),
        checkpoint_dir=os.path.join('checkpoints')
    )

    # Train model
    print("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs']
    )

    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Metrics: {test_metrics}")


if __name__ == "__main__":
    main()