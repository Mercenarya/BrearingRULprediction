# Model Workflow Detailed Documentation

## Overview
This document provides a detailed description of the bearing RUL prediction model workflow, including data preprocessing, feature extraction, temporal modeling, and prediction steps.

## 1. Data Preprocessing Pipeline
### 1.1 Raw Data Loading
- Vibration signals: 3-axis accelerometer data at 12 kHz sampling rate
- Temperature signals: Single-channel temperature measurements
- Loading format: CSV or MAT files from PRONOSTIA and XJTU-SY datasets

### 1.2 Signal Processing
1. **Bandpass Filtering**: 
   - Cutoff frequencies: 100 Hz - 5 kHz
   - Filter type: 4th order Butterworth
   - Purpose: Remove noise and irrelevant frequency components

2. **Normalization**:
   - Methods: Standardization (z-score) or Min-Max scaling
   - Applied per channel to ensure consistent scale

3. **Downsampling** (Optional):
   - Target sampling rate: Configurable (e.g., 4 kHz)
   - Method: Decimation with anti-aliasing filter

### 1.3 Sliding Window Segmentation
- Window size: 1024 samples (configurable)
- Overlap: 50% (configurable)
- Result: Tensor of shape [N, window_size, n_channels] where N is number of windows

## 2. Feature Extraction Stage
### 2.1 Multi-Scale CNN Encoding
Each modality (vibration and temperature) passes through a CNN encoder with three parallel branches:

#### Early Scale (Small Kernel):
- Kernel size: 3
- Purpose: Capture fine-grained local patterns
- Output: [batch_size, out_channels, seq_len]

#### Mid Scale (Medium Kernel):
- Kernel size: 5
- Purpose: Capture medium-range patterns
- Output: [batch_size, out_channels, seq_len]

#### Late Scale (Large Kernel):
- Kernel size: 7
- Purpose: Capture long-range patterns
- Output: [batch_size, out_channels, seq_len]

#### Feature Concatenation:
- Combined output: [batch_size, 3*out_channels, seq_len]
- Followed by transpose to [batch_size, seq_len, 3*out_channels] for subsequent layers

## 3. Cross-Modal Fusion
### 3.1 Cross-Modal Attention Mechanism
- Vibration features query temperature keys/values
- Temperature features query vibration keys/values
- Multi-head attention with 4 heads
- Output: Attended features for both modalities

### 3.2 Fusion Strategies
#### Early Fusion:
- Concatenate raw sensor inputs before processing

#### Mid Fusion (Implemented):
- Process modalities separately through CNN encoders
- Apply cross-modal attention
- Concatenate attended features
- Pass through fusion layer

#### Late Fusion:
- Process modalities completely separately
- Fuse at prediction level

#### Concat Fusion (Baseline):
- Simple concatenation without attention

## 4. Temporal Modeling
### 4.1 Dual-Path LSTM Architecture
#### Fast Path:
- Captures short-term dynamics and transient behaviors
- LSTM with configurable hidden size and layers
- Sensitive to recent changes in signal characteristics

#### Slow Path:
- Captures long-term trends and degradation patterns
- LSTM with configurable hidden size and layers
- Sensitive to gradual deterioration over time

#### Information Fusion:
- Concatenate hidden states from both paths
- Pass through combination layer
- Output: Integrated temporal features

## 5. Prediction Stage
### 5.1 Regression Head
Three-layer fully connected network:
- Layer 1: 192 → 128 neurons + BatchNorm + ReLU + Dropout
- Layer 2: 128 → 64 neurons + BatchNorm + ReLU + Dropout
- Layer 3: 64 → 1 neuron (RUL prediction)

### 5.2 Output Processing
- Raw network output represents predicted RUL
- Post-processing may include:
  - Clipping to valid RUL range [0, max_RUL]
  - Smoothing for sequential predictions
  - Unnormalization if normalized RUL was used during training

## 6. Loss Function
### 6.1 Stage-Aware PHM Loss
Asymmetric loss function that accounts for different error costs in prognostic stages:

#### Early Stage (High RUL):
- Under-prediction penalized more severely (conservative maintenance)
- Formula: L = exp(α * (y_true - y_pred)) - 1 for y_true > θ

#### Late Stage (Low RUL):
- Over-prediction penalized more severely (avoid unexpected failure)
- Formula: L = exp(-β * (y_pred - y_true)) - 1 for y_true ≤ θ

Where:
- α, β: Weighting parameters
- θ: Threshold separating early/late stages
- y_true: True RUL
- y_pred: Predicted RUL

## 7. Training Procedure
### 7.1 Optimization
- Algorithm: Adam optimizer
- Learning rate: Configurable (default: 0.001)
- Batch size: Configurable (default: 64)

### 7.2 Regularization
- Dropout: Applied in CNN, LSTM, and FC layers
- Batch normalization: Applied in CNN and FC layers
- Early stopping: Based on validation loss

### 7.3 Learning Rate Schedule
- ReduceLROnPlateau scheduler
- Factor: 0.5
- Patience: 5 epochs
- Minimum lr: 1e-6

## 8. Evaluation Metrics
### 8.1 Standard Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R-squared (R²)

### 8.2 Prognostic-Specific Metric
- PHM Score: Asymmetric error penalty metric from PHM 2012 Data Challenge

## 9. Implementation Details
### 9.1 Framework
- PyTorch 1.9+
- Python 3.8+

### 9.2 Hardware Requirements
- GPU: NVIDIA GTX 1080 Ti or better (recommended)
- CPU: Intel i7 or equivalent
- RAM: 16 GB minimum

### 9.3 Reproducibility
- Fixed random seeds for reproducible results
- Detailed logging of hyperparameters and metrics