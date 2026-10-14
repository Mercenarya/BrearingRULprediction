"""
Script to evaluate a trained model on multiple datasets for cross-dataset validation.
"""
import os,sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CURRENT_DIR, '..', '..')
sys.path.append(ROOT_DIR)
CONFIG = os.path.join(ROOT_DIR, 'config.yaml')

import torch
import yaml
import numpy as np
import argparse
from src.data.dataset import BearingDataset, create_data_loaders
from src.models.rul_model import RULModel
from src.evaluation.metrics import calculate_all_metrics


def load_config(config_path=CONFIG):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path, config):
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary

    Returns:
        Loaded model
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
    Evaluate model on a test data loader.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to use for evaluation

    Returns:
        Tuple: (predictions, truths, metrics)
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
    parser = argparse.ArgumentParser(description='Evaluate model on multiple datasets')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (e.g., checkpoints/best_model.pt)')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing .npz files')
    parser.add_argument('--config', type=str, default=CONFIG,
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cpu or cuda)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Define datasets to evaluate
    # We look for test files in the data directory
    test_files = [f for f in os.listdir(args.data_dir) if f.startswith('test_') and f.endswith('.npz')]
    # Also include the default test.npz if it exists
    if 'test.npz' in os.listdir(args.data_dir):
        test_files.append('test.npz')
    # Remove duplicates and sort
    test_files = sorted(list(set(test_files)))

    print(f"Found {len(test_files)} test datasets: {test_files}")

    # Load the model once
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, config)

    # Evaluate on each dataset
    all_results = {}
    for test_file in test_files:
        test_path = os.path.join(args.data_dir, test_file)
        print(f"\n{'='*50}")
        print(f"Evaluating on: {test_file}")
        print(f"{'='*50}")

        # We need to create a data loader for this test set.
        # Since create_data_loaders expects train, val, test paths, we can provide dummy paths for train and val
        # but we only care about the test loader.
        # Alternatively, we can create a BearingDataset and DataLoader directly.
        # Let's do the latter for simplicity.

        # Load the test dataset
        test_dataset = BearingDataset(test_path)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0
        )

        # Evaluate
        predictions, truths, metrics = evaluate_model(model, test_loader, device)

        # Store results
        all_results[test_file] = {
            'predictions': predictions,
            'truths': truths,
            'metrics': metrics
        }

        # Print metrics
        print(f"Metrics for {test_file}:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name.upper()}: {metric_value:.4f}")

    # Summary
    print(f"\n{'='*50}")
    print("CROSS-DATASET EVALUATION SUMMARY")
    print(f"{'='*50}")
    # Print a table of metrics
    if all_results:
        # Get metric names from the first result
        first_key = next(iter(all_results))
        metric_names = list(all_results[first_key]['metrics'].keys())
        # Print header
        header = "Dataset".ljust(20) + "".join([f"{m.upper():>12}" for m in metric_names])
        print(header)
        print("-" * len(header))
        for test_file, result in all_results.items():
            row = test_file.ljust(20)
            for m in metric_names:
                val = result['metrics'][m]
                row += f"{val:>12.4f}"
            print(row)

    # Save results
    results_dir = os.path.join('results', 'cross_dataset')
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics as YAML
    metrics_to_save = {}
    for test_file, result in all_results.items():
        metrics_to_save[test_file] = result['metrics']

    import yaml
    with open(os.path.join(results_dir, 'metrics.yaml'), 'w') as f:
        yaml.dump(metrics_to_save, f)

    # Optionally, save predictions and truths (can be large, so we might skip)
    # For now, we save only the first dataset's predictions as an example
    if len(all_results) > 0:
        first_file = next(iter(all_results))
        first_result = all_results[first_file]
        np.savez(
            os.path.join(results_dir, f'{first_file}_predictions.npz'),
            predictions=first_result['predictions'],
            truths=first_result['truths']
        )

    print(f"\nResults saved to: {results_dir}/")


if __name__ == "__main__":
    main()