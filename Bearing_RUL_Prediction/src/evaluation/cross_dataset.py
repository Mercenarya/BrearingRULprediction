"""
Cross-dataset evaluation script for Bearing RUL prediction.
Tests generalization between PRONOSTIA and XJTU-SY datasets.
"""
import os
import torch
import yaml
import numpy as np
from src.data.dataset import BearingDataset, create_data_loaders
from src.models.rul_model import RULModel
from src.training.trainer import Trainer
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

def evaluate_cross_dataset(train_on, test_on, config):
    """
    Evaluate model trained on one dataset and tested on another.

    Args:
        train_on (str): Dataset used for training ('pronostia' or 'xjtusy')
        test_on (str): Dataset used for testing ('pronostia' or 'xjtusy')
        config (dict): Configuration dictionary

    Returns:
        tuple: (metrics, predictions, truths)
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine data paths based on datasets
    data_dir = os.path.join('data', 'processed')
    if train_on == 'pronostia':
        train_path = os.path.join(data_dir, 'train_pronostia.npz')
    else:  # xjtusy
        train_path = os.path.join(data_dir, 'train_xjtusy.npz')

    if test_on == 'pronostia':
        test_path = os.path.join(data_dir, 'test_pronostia.npz')
    else:  # xjtusy
        test_path = os.path.join(data_dir, 'test_xjtusy.npz')

    # Check if files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")

    # Create data loaders
    _, _, test_loader = create_data_loaders(
        train_path=train_path,
        val_path=train_path,  # Use same as train for validation (simplified)
        test_path=test_path,
        batch_size=config['training']['batch_size'],
        num_workers=0
    )

    # Load trained model (assuming we have a pre-trained model)
    # In practice, you would train the model first
    model_path = os.path.join('checkpoints', f'best_model_{train_on}.pt')
    if not os.path.exists(model_path):
        # If pre-trained model doesn't exist, we'll train quickly
        print(f"No pre-trained model found for {train_on}. Training quick model...")

        # Create train loader
        train_loader, val_loader, _ = create_data_loaders(
            train_path=train_path,
            val_path=os.path.join(data_dir, 'val_pronostia.npz' if train_on == 'pronostia' else 'val_xjtusy.npz'),
            test_path=test_path,
            batch_size=config['training']['batch_size'],
            num_workers=0
        )

        # Create model
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
        model = model.to(device)

        # Setup loss
        loss_config = config.get('loss', {})
        criterion = StageAwarePHMLoss(
            alpha=loss_config.get('alpha', 0.5),
            beta=loss_config.get('beta', 0.3),
            theta=loss_config.get('theta', 0.1)
        )

        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

        # Setup trainer
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            early_stopping_patience=5,  # Shorter for quick training
            checkpoint_dir=os.path.join('checkpoints', f'temp_{train_on}'),
            use_wandb=False
        )

        # Train quickly
        trainer.train(train_loader, val_loader, num_epochs=10)

        # Save the model
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)
        print(f"Quick model saved to {model_path}")
    else:
        # Load pre-trained model
        model = load_model(model_path, config)
        model = model.to(device)

    # Setup loss function for evaluation
    loss_config = config.get('loss', {})
    criterion = StageAwarePHMLoss(
        alpha=loss_config.get('alpha', 0.5),
        beta=loss_config.get('beta', 0.3),
        theta=loss_config.get('theta', 0.1)
    )

    # Evaluate
    test_loss, test_metrics = Trainer.evaluate_nonexistent(model, criterion, test_loader, device)

    # Since we don't have access to the trainer's evaluate method directly,
    # we'll implement evaluation here
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            vib_inputs = batch['vibration'].to(device)
            temp_inputs = batch['temperature'].to(device)
            labels = batch['rul'].to(device).unsqueeze(1)

            outputs = model(vib_inputs, temp_inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = calculate_all_metrics(all_labels, all_preds)
    metrics['loss'] = avg_loss

    return metrics, all_preds, all_labels

def run_cross_dataset_evaluation():
    """
    Run cross-dataset evaluation between PRONOSTIA and XJTU-SY.
    """
    print("Cross-Dataset Evaluation: PRONOSTIA ↔ XJTU-SY")
    print("=" * 50)

    # Load configuration
    config = load_config()

    # Define dataset combinations
    datasets = ['pronostia', 'xjtusy']
    results = {}

    for train_on in datasets:
        for test_on in datasets:
            key = f"{train_on}→{test_on}"
            print(f"\nEvaluating: Trained on {train_on.upper()}, Tested on {test_on.upper()}")

            try:
                metrics, predictions, truths = evaluate_cross_dataset(train_on, test_on, config)
                results[key] = {
                    'metrics': metrics,
                    'predictions_shape': predictions.shape,
                    'truths_shape': truths.shape
                }

                print(f"  Loss: {metrics['loss']:.4f}")
                print(f"  MAE: {metrics['mae']:.4f}")
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  R²: {metrics['r2']:.4f}")
                print(f"  PHM Score: {metrics['phm_score']:.4f}")

            except Exception as e:
                print(f"  Error: {e}")
                results[key] = {'error': str(e)}

    # Print summary table
    print(f"\n{'='*60}")
    print("CROSS-DATASET EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Train→Test':<15} {'PHM Score':<12} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
    print("-" * 65)
    for train_on in datasets:
        for test_on in datasets:
            key = f"{train_on}→{test_on}"
            if 'error' not in results[key]:
                m = results[key]['metrics']
                print(f"{key:<15} {m['phm_score']:<12.4f} {m['mae']:<12.4f} "
                      f"{m['rmse']:<12.4f} {m['r2']:<12.4f}")
            else:
                print(f"{key:<15} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")

    # Save results
    results_dir = os.path.join('results', 'tables')
    os.makedirs(results_dir, exist_ok=True)

    import json
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if 'error' not in value:
            json_results[key] = {
                'metrics': value['metrics'],
                'shapes': {
                    'predictions': list(value['predictions_shape']),
                    'truths': list(value['truths_shape'])
                }
            }
        else:
            json_results[key] = {'error': value['error']}

    with open(os.path.join(results_dir, 'cross_dataset_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nDetailed results saved to {results_dir}/cross_dataset_results.json")

# Helper method for evaluation (since we can't modify Trainer class easily)
def evaluate_nonexistent(model, criterion, data_loader, device):
    """
    Helper function to evaluate model (placeholder since we can't modify Trainer).
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            vib_inputs = batch['vibration'].to(device)
            temp_inputs = batch['temperature'].to(device)
            labels = batch['rul'].to(device).unsqueeze(1)

            outputs = model(vib_inputs, temp_inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # This is a simplified version - in practice you'd return proper metrics
    from src.evaluation.metrics import calculate_all_metrics
    metrics = calculate_all_metrics(all_labels, all_preds)
    metrics['loss'] = avg_loss

    return avg_loss, metrics

if __name__ == "__main__":
    run_cross_dataset_evaluation()