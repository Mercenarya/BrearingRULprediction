# Training and Evaluation Skill for Bearing RUL Prediction

This skill covers model training, evaluation, and experimentation workflows.

## Model Architecture
The bearing RUL prediction model consists of:
- **Multi-scale CNN Encoder**: 3 parallel branches with kernel sizes [3, 5, 7]
- **Cross-Modal Attention**: 4-head attention between vibration and temperature modalities
- **Dual-Path LSTM**: 
  - Fast path: Captures short-term dynamics
  - Slow path: Captures long-term trends
- **Regression Head**: 3-layer FC network (192 → 128 → 64 → 1)

## Configuration
Model and training parameters are controlled via `config.yaml`:
- Data: window_size, overlap, normalization
- Model: CNN channels, fusion type, LSTM size, attention heads
- Training: batch_size, learning_rate, num_epochs, optimizer
- Logging: W&B integration, project name

## Training
Standard training execution:
```bash
python src/training/train.py
```
Features:
- Automatic loading of train/val/test data from data/processed/
- Weights & Biases logging (when enabled)
- Checkpoint saving: best_model.pt (best val loss) and last_model.pt (latest epoch)
- Test set evaluation after training

## Evaluation
### Standard Evaluation
Evaluate on primary test set:
```bash
python src/evaluation/evaluate.py
```
Computes: MAE, RMSE, R², PHM Score

### Cross-Dataset Evaluation
Test generalization across bearing datasets:
```bash
python src/evaluation/cross_dataset_evaluate.py --checkpoint checkpoints/best_model.pt
```
Evaluates on:
- test.npz (primary dataset)
- test_pronostia.npz (domain shift test)
- test_xjtusy.npz (alternative split)
Outputs comparison table showing performance degradation/gain

## Ablation Studies
Run ablation studies to validate component effectiveness:
```bash
python src/evaluation/ablation.py
```
Tests impact of:
- Multi-scale CNN vs single-scale
- Cross-modal attention vs concatenation
- Dual-path LSTM vs single-path
- Stage-aware loss vs symmetric loss

## Usage
Use this skill when you need to:
- Train new model configurations
- Evaluate model performance
- Compare experiments via W&B
- Validate generalization across datasets
- Conduct ablation studies for paper