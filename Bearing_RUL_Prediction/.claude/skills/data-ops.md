# Data Operations Skill for Bearing RUL Prediction

This skill covers data handling operations for the bearing RUL prediction project.

## Data Validation
Verify integrity of processed .npz files:
- Checks for required keys: vibration, temperature, rul
- Validates data types (should be float32)
- Confirms shapes: 
  - vibration: (n_samples, 1024, 2) 
  - temperature: (n_samples, 1024, 1)
  - rul: (n_samples,)
- Ensures no NaN or Inf values

Command: `python src/data/validate_data.py`

## Data Versioning
Maintain dataset registry tracking:
- Filename
- Source (Pronostia/XJTU-SY/Unknown)
- Version
- Preprocessing steps
- SHA256 checksum
- File size
- Date added

Command: `python src/data/generate_registry.py`
Output: Updates data/processed/registry.csv

## Data Sources
The project supports two main bearing datasets:
1. **PRONOSTIA** - Bearing acceleration data 
2. **XJTU-SY** - Xi'an Jiaotong University bearing dataset

Files are typically organized as:
- train.npz, val.npz, test.npz (primary dataset, often XJTU-SY)
- train_pronostia.npz, val_pronostia.npz, test_pronostia.npz
- train_xjtusy.npz, val_xjtusy.npz, test_xjtusy.npz

## Preprocessing Pipeline (if working with raw data)
If raw data needs processing:
1. Download: `python src/data/download.py` (from Kaggle)
2. Preprocess: `python src/data/preprocessing.py` (bandpass filter, downsample, normalize)
3. Sliding windows: `python src/data/sliding_window.py` (if needed)
4. Output: .npz files in data/processed/ with standard structure

## Usage
Use this skill when you need to:
- Validate data before training
- Version datasets for reproducibility  
- Prepare new data for experiments
- Troubleshoot data-related issues