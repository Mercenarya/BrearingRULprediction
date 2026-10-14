# Bearing RUL Prediction Project Guidelines

## Project Overview
This project develops a deep learning approach for predicting the Remaining Useful Life (RUL) of bearings using multi-modal sensor data (vibration and temperature). The solution uses:
- Multi-scale CNN Encoder (kernel sizes [3,5,7])
- Cross-Modal Attention Mechanism 
- Dual-Path LSTM Architecture (fast/slow paths)
- Stage-Aware PHM Loss Function

## Key Technical Details
- Input: Vibration (3-axis, 12kHz) + Temperature (single channel)
- Window size: 1024 samples with 50% overlap
- Feature dimension after CNN: 192 channels per modality
- LSTM hidden size: 128 (each path)
- Regression head: 192 → 128 → 64 → 1
- Datasets: PRONOSTIA and XJTU-SY

## ML Ops Priorities (from ML_Ops_Obj.md)
1. **Data Validation** - Validate .npz files have correct keys, dtypes, shapes
2. **Data Versioning** - Maintain registry.csv with file metadata
3. **Pipeline Smoke Test** - Run training with 2 epochs to verify pipeline integrity
4. **Experiment Tracking** - Enable W&B logging via config.yaml
5. **Cross-Dataset Evaluation** - Test generalization across PRONOSTIA ↔ XJTU-SY

## Standard Workflow
1. Validate data: `python src/data/validate_data.py`
2. Generate registry: `python src/data/generate_registry.py`  
3. Configure training: Edit config.yaml (adjust epochs, batch size, W&B)
4. Train model: `python src/training/train.py`
5. Evaluate: `python src/evaluation/evaluate.py`
6. Cross-dataset eval: `python src/evaluation/cross_dataset_evaluate.py --checkpoint checkpoints/best_model.pt`

## File Structure Key Locations
- Data: `data/processed/` (.npz files)
- Source: `src/` (data, models, losses, training, evaluation)
- Notebooks: `notebooks/` (EDA and results analysis)
- Config: `config.yaml`
- Checkpoints: `checkpoints/`
- Results: `results/` (expected but not yet created)