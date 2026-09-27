"""
Script to evaluate trained models on test set.
"""
import os
import torch
import yaml
import numpy as np
from src.data.dataset import BearingDataset, create_data_loaders
from src.models.rul_model import RULModel
from src.evaluation.metrics import calculate_all_metrics

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

def load_model(checkpoint_path, config):
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path (str): Path to model checkpoint
        config (dict): Configuration dictionary

    Returns:
        nn.Module: Loaded model
    """
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

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model on test set.

    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        device (str): Device to use for evaluation

    Returns:
        tuple: (predictions, truths, metrics)
    """
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # Get data
            vib_inputs = batch['vibration'].to(device)
            temp_inputs = batch['temperature'].to(device)
            labels = batch['rul'].to(device)

            # Forward pass
            outputs = model(vib_inputs, temp_inputs)

            # Store results
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_preds, axis=0)
    truths = np.concatenate(all_labels, axis=0)

    # Calculate metrics
    metrics = calculate_all_metrics(truths, predictions)

    return predictions, truths, metrics

def main():
    """
    Main evaluation function.
    """
    # Load configuration
    config = load_config()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create test data loader
    _, _, test_loader = create_data_loaders(
        train_path=os.path.join('data', 'processed', 'train.npz'),
        val_path=os.path.join('data', 'processed', 'val.npz'),
        test_path=os.path.join('data', 'processed', 'test.npz'),
        batch_size=config['testing']['batch_size'] if 'testing' in config else config['training']['batch_size'],
        num_workers=0
    )

    # Load trained model
    model_path = os.path.join('checkpoints', 'best_model.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print(f"Loading model from {model_path}")
    model = load_model(model_path, config)

    # Evaluate model
    print("Evaluating model on test set...")
    predictions, truths, metrics = evaluate_model(model, test_loader, device)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper()}: {metric_value:.4f}")

    # Save results
    results_dir = os.path.join('results', 'tables')
    os.makedirs(results_dir, exist_ok=True)

    # Save predictions and truths
    np.savez(
        os.path.join(results_dir, 'test_predictions.npz'),
        predictions=predictions,
        truths=truths
    )

    # Save metrics as YAML
    import yaml
    with open(os.path.join(results_dir, 'test_metrics.yaml'), 'w') as f:
        yaml.dump(metrics, f)

    print(f"\nResults saved to {results_dir}/")

if __name__ == "__main__":
    main()