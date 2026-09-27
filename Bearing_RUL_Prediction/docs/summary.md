# Project Summary: Bearing RUL Prediction

## Project Overview
This project develops a deep learning approach for predicting the Remaining Useful Life (RUL) of bearings using multi-modal sensor data (vibration and temperature). The solution addresses key challenges in prognostic modeling including feature extraction from raw signals, multi-modal fusion, temporal dependency modeling, and asymmetric error penalization.

## Key Innovations
1. **Multi-scale CNN Encoder**: Extracts features at early, mid, and late scales to capture patterns of different temporal resolutions
2. **Cross-Modal Attention Mechanism**: Enables vibration and temperature modalities to inform each other's feature representation
3. **Dual-Path LSTM Architecture**: Separately models short-term dynamics (fast path) and long-term trends (slow path)
4. **Stage-Aware PHM Loss Function**: Implements asymmetric error weighting based on prognostic severity levels

## Technical Specifications

### Input Data
- Vibration signals: 3-axis acceleration, 12 kHz sampling rate
- Temperature signals: Single channel, synchronized with vibration
- Supported datasets: PRONOSTIA, XJTU-SY

### Model Architecture
- CNN Encoder: 3 parallel branches with kernel sizes [3, 5, 7]
- Feature dimension after CNN: 3 × 64 = 192 channels per modality
- Attention heads: 4
- LSTM hidden size: 128 (each path)
- Regression head: 192 → 128 → 64 → 1
- Dropout rate: 0.3

### Training Configuration
- Optimizer: Adam (learning rate = 0.001)
- Batch size: 64
- Epochs: 100 (with early stopping)
- Loss function: Stage-aware PHM loss (α=0.5, β=0.3, θ=0.1)

### Evaluation Metrics
- Primary: PHM Score (lower is better)
- Secondary: MAE, RMSE, R²
- Cross-validation: Hold-out test set
- Generalization: Cross-dataset testing (PRONOSTIA ↔ XJTU-SY)

## File Structure
```
Bearing_RUL_Prediction/
├── data/                    # Data storage (raw, processed, external)
├── src/                     # Source code
│   ├── data/                # Data loading and preprocessing
│   ├── models/              # Model architectures
│   ├── losses/              # Loss functions
│   ├── training/            # Training procedures
│   ├── evaluation/          # Model evaluation
│   └── utils/               # Utility functions
├── notebooks/               # Jupyter notebooks for analysis
├── paper/                   # Manuscript and references
├── docs/                    # Documentation
├── checkpoints/             # Saved model weights
├── results/                 # Outputs, figures, tables, logs
└── tests/                   # Unit tests
```

## Usage Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Download data: `python src/data/download.py`
3. Preprocess data: `python src/data/preprocessing.py`
4. Create sliding windows: `python src/data/sliding_window.py`
5. Train model: `python src/training/train.py`
6. Evaluate model: `python src/evaluation/evaluate.py`
7. Run ablation study: `python src/evaluation/ablation.py`
8. Test cross-dataset generalization: `python src/evaluation/cross_dataset.py`

## Expected Outcomes
- RUL prediction with PHM Score < 0.05 (achievable target)
- MAE < 5.0% of maximum RUL
- Strong generalization across different bearing datasets
- Interpretable feature importance through attention weights

## Contributors
- Person A: Data preprocessing, feature extraction, CNN encoder design
- Person B: Temporal modeling, loss function design, training procedures  
- Person C: Model evaluation, visualization, experimental analysis, paper writing

## Future Work
1. Incorporate additional sensor modalities (acoustic, current)
2. Implement uncertainty quantification in predictions
3. Extend to multi-component systems and fault isolation
4. Deploy as embedded system for real-time monitoring
5. Explore transformer-based alternatives to LSTM