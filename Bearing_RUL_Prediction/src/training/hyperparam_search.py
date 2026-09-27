"""
Hyperparameter search for learning rate, batch size, and alpha parameter.
"""
import itertools
import yaml
import os
import subprocess
import sys

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

def save_config(config, config_path="config.yaml"):
    """
    Save configuration to YAML file.

    Args:
        config (dict): Configuration dictionary
        config_path (str): Path to save configuration file
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def run_training(config):
    """
    Run training with given configuration.

    Args:
        config (dict): Configuration dictionary

    Returns:
        float: Validation loss from the training run
    """
    # Save temporary config
    temp_config_path = "temp_config.yaml"
    save_config(config, temp_config_path)

    # Run training script
    cmd = [sys.executable, "-m", "src.training.train"]
    env = os.environ.copy()
    env["CONFIG_PATH"] = temp_config_path

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode != 0:
            print(f"Training failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return float('inf')

        # Extract validation loss from output (simplified)
        # In practice, you'd want to parse logs or return values properly
        lines = result.stdout.split('\n')
        val_loss = float('inf')
        for line in lines:
            if "Val Loss:" in line:
                try:
                    val_loss = float(line.split("Val Loss:")[1].split(",")[0].strip())
                except:
                    pass
                break

        return val_loss

    except subprocess.TimeoutExpired:
        print("Training timed out")
        return float('inf')
    except Exception as e:
        print(f"Error running training: {e}")
        return float('inf')
    finally:
        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def grid_search_lr_batch_alpha():
    """
    Perform grid search over learning rate, batch size, and alpha.
    """
    # Load base configuration
    config = load_config()

    # Define search space
    learning_rates = [0.0001, 0.001, 0.01]
    batch_sizes = [32, 64, 128]
    alpha_values = [0.1, 0.3, 0.5, 0.7]

    best_loss = float('inf')
    best_params = None

    total_combinations = len(learning_rates) * len(batch_sizes) * len(alpha_values)
    current = 0

    print(f"Starting grid search over {total_combinations} combinations...")

    for lr, batch_size, alpha in itertools.product(learning_rates, batch_sizes, alpha_values):
        current += 1
        print(f"\nCombination {current}/{total_combinations}:")
        print(f"  Learning rate: {lr}")
        print(f"  Batch size: {batch_size}")
        print(f"  Alpha: {alpha}")

        # Update configuration
        config['training']['learning_rate'] = lr
        config['training']['batch_size'] = batch_size
        config['loss']['alpha'] = alpha

        # Run training
        val_loss = run_training(config)

        print(f"  Validation Loss: {val_loss:.4f}")

        # Update best parameters
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = {
                'learning_rate': lr,
                'batch_size': batch_size,
                'alpha': alpha
            }
            print(f"  *** New best loss: {best_loss:.4f} ***")

    print("\n" + "="*50)
    print("Grid Search Completed!")
    print(f"Best Validation Loss: {best_loss:.4f}")
    print(f"Best Parameters: {best_params}")
    print("="*50)

    # Save best configuration
    if best_params:
        config['training']['learning_rate'] = best_params['learning_rate']
        config['training']['batch_size'] = best_params['batch_size']
        config['loss']['alpha'] = best_params['alpha']
        save_config(config, "best_config.yaml")
        print("Best configuration saved to best_config.yaml")

def random_search(num_trials=20):
    """
    Perform random search over hyperparameter space.

    Args:
        num_trials (int): Number of random trials to perform
    """
    import random

    # Load base configuration
    config = load_config()

    # Define search distributions
    lr_range = (-4, -1)  # log10 range: 0.0001 to 0.1
    batch_size_options = [16, 32, 64, 128, 256]
    alpha_range = (0.05, 0.9)  # alpha range

    best_loss = float('inf')
    best_params = None

    print(f"Starting random search with {num_trials} trials...")

    for trial in range(num_trials):
        # Sample random hyperparameters
        lr = 10**random.uniform(*lr_range)
        batch_size = random.choice(batch_size_options)
        alpha = random.uniform(*alpha_range)

        print(f"\nTrial {trial+1}/{num_trials}:")
        print(f"  Learning rate: {lr:.6f}")
        print(f"  Batch size: {batch_size}")
        print(f"  Alpha: {alpha:.4f}")

        # Update configuration
        config['training']['learning_rate'] = lr
        config['training']['batch_size'] = batch_size
        config['loss']['alpha'] = alpha

        # Run training
        val_loss = run_training(config)

        print(f"  Validation Loss: {val_loss:.4f}")

        # Update best parameters
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = {
                'learning_rate': lr,
                'batch_size': batch_size,
                'alpha': alpha
            }
            print(f"  *** New best loss: {best_loss:.4f} ***")

    print("\n" + "="*50)
    print("Random Search Completed!")
    print(f"Best Validation Loss: {best_loss:.4f}")
    print(f"Best Parameters: {best_params}")
    print("="*50)

    # Save best configuration
    if best_params:
        config['training']['learning_rate'] = best_params['learning_rate']
        config['training']['batch_size'] = best_params['batch_size']
        config['loss']['alpha'] = best_params['alpha']
        save_config(config, "best_config.yaml")
        print("Best configuration saved to best_config.yaml")

if __name__ == "__main__":
    print("Hyperparameter Search Module")
    print("Choose search type:")
    print("1. Grid search over LR, batch size, alpha")
    print("2. Random search")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        grid_search_lr_batch_alpha()
    elif choice == "2":
        num_trials = int(input("Number of random trials: ") or "20")
        random_search(num_trials)
    else:
        print("Invalid choice. Running grid search by default.")
        grid_search_lr_batch_alpha()