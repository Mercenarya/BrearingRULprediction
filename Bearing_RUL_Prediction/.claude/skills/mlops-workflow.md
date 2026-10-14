# MLOps Workflow Skill for Bearing RUL Prediction

This skill encapsulates the standard MLOps workflow for the bearing RUL prediction project.

## Workflow Steps

### 1. Data Validation

Validate all .npz files in data/processed/ have correct structure:

```bash
python src/data/validate_data.py
```

Expected output: All files show [VALID]

### 2. Data Versioning

Generate or update the dataset registry:

```bash
python src/data/generate_registry.py
```

Expected output: Registry written to registry.csv with file metadata

### 3. Pipeline Smoke Test

Verify training pipeline works with minimal epochs:

1. Temporarily modify config.yaml:
   - training.num_epochs: 2
   - training.batch_size: 16 (reduce if needed)
   - logging.use_wandb: false
2. Run training:

```bash
python src/training/train.py
```

Expected output: Training completes 2 epochs and evaluates on test set

### 4. Experiment Tracking

Enable Weights & Biases for experiment tracking:

1. Set in config.yaml: logging.use_wandb: true
2. Optionally set project name: logging.project_name: "your-project-name"
3. Install W&B if needed: pip install wandb
4. Login: wandb login
5. Run full training: python src/training/train.py

### 5. Cross-Dataset Evaluation

Test model generalization across datasets:

```bash
python src/evaluation/cross_dataset_evaluate.py --checkpoint checkpoints/best_model.pt
```

Expected output: Comparison table showing performance on test.npz, test_pronostia.npz, test_xjtusy.npz

## Usage

Invoke this skill to get reminders about the standard workflow or to check if you've completed all MLOps prerequisites before experimentation.
