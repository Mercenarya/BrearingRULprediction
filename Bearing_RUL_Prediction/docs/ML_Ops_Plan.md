# ML Ops Plan and Workflow for Bearing RUL Prediction

## 1. Data Management

### 1.1 Data Storage
- Store normalized datasets in `data/processed/` directory.
- Current datasets:
  - `train.npz`, `val.npz`, `test.npz`: Primary dataset (likely XJTU-SY)
  - `train_pronostia.npz`, `val_pronostia.npz`, `test_pronostia.npz`: Pronostia dataset
  - `train_xjtusy.npz`, `val_xjtusy.npz`, `test_xjtusy.npz`: XJTU-SY dataset (alternative split)
- Each `.npz` file contains three arrays:
  - `vibration`: Shape `(n_samples, window_size, 2)` (float32)
  - `temperature`: Shape `(n_samples, window_size, 1)` (float32)
  - `rul`: Shape `(n_samples,)` (float32)

### 1.2 Data Versioning
- Maintain a `data/registry.csv` (or JSON) to track:
  - Dataset name
  - Source (e.g., XJTU-SY, Pronostia)
  - Version/date
  - Preprocessing steps applied (bandpass filter, downsampling, normalization)
  - File checksum (MD5 or SHA256) for integrity
- Update registry when new datasets are added.

### 1.3 Data Validation
- Implement a validation script to check:
  - File existence and readability
  - Required keys (`vibration`, `temperature`, `rul`)
  - Data types (float32)
  - Shape consistency (window size 1024, channels as expected)
  - Value ranges (no NaNs or infs)

## 2. Pipeline Setup

### 2.1 Training Pipeline
- Use existing script: `src/training/train.py`
- Configuration via `config.yaml`:
  - Data parameters: sampling_rate, window_size, overlap, normalization
  - Model parameters: CNN scales, fusion type, LSTM size, etc.
  - Training parameters: batch size, learning rate, epochs, optimizer
- The script automatically:
  - Loads data from `data/processed/train.npz`, `val.npz`, `test.npz`
  - Creates data loaders via `src/data/dataset.py`
  - Trains model with Weights & Biases logging (if enabled)
  - Saves best model to `checkpoints/best_model.pt`
  - Evaluates on test set and logs metrics

### 2.2 Evaluation Pipeline
- Extend existing evaluation script (`src/evaluation/evaluate.py`) to support cross-dataset evaluation.
- Create a new script `src/evaluation/cross_dataset_evaluate.py` that:
  - Loads a trained model checkpoint
  - Evaluates on multiple datasets (primary, Pronostia, XJTU-SY)
  - Computes and compares metrics across datasets
  - Generates a report (table/plot) showing performance degradation/gain

### 2.3 Preprocessing Pipeline (if raw data is used)
- Though we are using normalized data, the raw data preprocessing pipeline is available in:
  - `src/data/download.py`: Downloads from Kaggle
  - `src/data/preprocessing.py`: Applies bandpass filter, downsamples, normalizes
  - `src/data/sliding_window.py`: Creates sliding windows (if needed)
- If raw data is obtained, run:
  ```bash
  python src/data/download.py
  python src/data/preprocessing.py
  ```
- Outputs should be placed in `data/processed/` with the same `.npz` structure.

## 3. Experiment Tracking

### 3.1 Weights & Biases Integration
- Enable in `config.yaml`: `logging.use_wandb: true`
- Configure project name: `logging.project_name`
- Each training run logs:
  - Hyperparameters (from config)
  - Metrics per epoch (loss, MAE, RMSE, etc.)
  - Model artifacts (checkpoints)
  - System metrics (GPU usage, etc.)
- Use W&B dashboard to compare experiments.

### 3.2 Experiment Reproducibility
- Tag each experiment with:
  - Git commit hash
  - Dataset version (from registry)
  - Preprocessing configuration
- Store final model and metadata in W&B artifacts.

## 4. Model Versioning

### 4.1 Checkpoint Management
- Training script saves:
  - `checkpoints/last_model.pt`: Latest epoch
  - `checkpoints/best_model.pt`: Best validation loss
- Include in checkpoint:
  - Model state dict
  - Optimizer state dict
  - Epoch
  - Validation loss
  - Configuration used

### 4.2 Model Registry (Future)
- For production, consider using MLflow or W&B Model Registry to:
  - Stage models (Staging, Production)
  - Track model lineage
  - Enable model rollback

## 5. Deployment Preparation

### 5.1 Model Export
- Export trained model to TorchScript or ONNX for serving:
  ```python
  # Example TorchScript export
  model.eval()
  traced_model = torch.jit.trace(model, (vib_example, temp_example))
  traced_model.save("model_traced.pt")
  ```

### 5.2 Inference Template
- Create a simple inference service (e.g., FastAPI) that:
  - Loads the exported model
  - Accepts vibration and temperature arrays
  - Returns RUL prediction
  - Includes input validation and error handling

### 5.3 Dependency Management
- Export requirements for deployment:
  ```bash
  pip freeze > requirements_deployment.txt
  ```
- Ensure compatibility with target environment (CPU/GPU, Python version).

## 6. Monitoring and Maintenance

### 6.1 Training Monitoring
- Monitor training logs for:
  - Loss convergence
  - Overfitting (train vs. val loss)
  - Anomalies (NaN losses, exploding gradients)
- Set up alerts (via W&B or custom) for training failures.

### 6.2 Model Monitoring (Post-deployment)
- Track prediction drift:
  - Compare predicted RUL distribution with expected
  - Monitor input data statistics (vibration, temperature) for shifts
- Schedule periodic re-evaluation on latest data.

### 6.3 Data Pipeline Monitoring
- Validate incoming data schema and quality
- Alert on missing data or preprocessing failures

## 7. Workflow

### Step 1: Data Preparation
1. Copy normalized `.npz` files to `data/processed/`:
   ```bash
   cp /path/to/dataset/*.npz E:/BearingRULprediction/Bearing_RUL_Prediction/data/processed/
   ```
2. Run data validation script (to be created) to verify integrity.
3. Update `data/registry.csv` with dataset information.

### Step 2: Configuration
1. Review and adjust `config.yaml` as needed (e.g., batch size, learning rate).
2. Ensure `logging.use_wandb: true` for experiment tracking.
3. Set `data.normalization` to match the preprocessing applied (should be "standard").

### Step 3: Training
1. Execute training:
   ```bash
   python src/training/train.py
   ```
2. Monitor progress via:
   - Console output
   - Weights & Biases dashboard (if enabled)
3. Training outputs:
   - Checkpoints in `checkpoints/`
   - Final metrics logged to W&B and console

### Step 4: Evaluation
1. Evaluate on primary test set (already done in training script).
2. Run cross-dataset evaluation:
   ```bash
   python src/evaluation/cross_dataset_evaluate.py --checkpoint checkpoints/best_model.pt
   ```
3. Analyze results:
   - Primary dataset: Expected performance
   - Cross-dataset: Check for generalization gaps

### Step 5: Model Selection and Promotion
1. Select best model based on:
   - Primary dataset validation/test performance
   - Cross-dataset robustness (if required)
2. Promote model to production:
   - Copy `checkpoints/best_model.pt` to `models/production/`
   - Export to TorchScript/ONNX for deployment
   - Update model registry (if applicable)

### Step 6: Documentation and Reporting
1. Update `README.md` with:
   - Instructions to reproduce results
   - Dataset descriptions and sources
   - Model performance summary
2. Generate a report (PDF/HTML) summarizing:
   - Experiments conducted
   - Best model performance
   - Comparative analysis across datasets
   - Deployment considerations

### Step 7: Maintenance
1. Schedule periodic retraining (e.g., monthly) when new data arrives.
2. Monitor production model for data drift and performance degradation.
3. Update dataset registry and preprocessing steps as needed.

## 8. Tools and Technologies
- **Framework**: PyTorch
- **Experiment Tracking**: Weights & Biases (integrated)
- **Data Validation**: Custom scripts (using NumPy)
- **Model Export**: TorchScript, ONNX
- **Deployment**: FastAPI/Docker (template to be created)
- **Version Control**: Git (for code), DVC or MLflow (recommended for data/models)

## 9. Next Steps
1. Implement data validation script.
2. Create cross-dataset evaluation script.
3. Create model export and inference template.
4. Set up automated retraining pipeline (e.g., using GitHub Actions or cron).
5. Document all procedures in the project's `docs/` directory.

---
*Plan created on: 2026-09-30*
*For: ML Ops Role - Bearing RUL Prediction Project*