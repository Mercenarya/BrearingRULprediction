# Report Generation Skill for Bearing RUL Prediction

This skill covers generating reports, visualizations, and artifacts for the bearing RUL prediction project.

## Paper Figure Generation
The notebooks contain code for generating publication-ready figures:
- **01_EDA.ipynb**: Exploratory data analysis visualizations
- **03_results_analysis.ipynb**: Final results and paper figures (Fig 1-8)

Figure characteristics:
- Style: publication-ready (seaborn-whitegrid, paper context)
- Dimensions: 8x6 inches
- DPI: 300
- Font sizes: optimized for academic papers
- Color palette: colorblind-friendly

## Results Analysis
Generate analysis artifacts from notebooks:
1. Training curves visualization
2. Prediction vs Actual plots  
3. Error distribution analysis
4. Ablation study results visualization
5. Cross-dataset generalization results
6. Feature importance analysis
7. Paper figure generation (referenced in 03_results_analysis.ipynb)

## Experiment Tracking Artifacts
When using Weights & Biases:
- Automatic logging of hyperparameters from config.yaml
- Per-epoch metrics (loss, MAE, RMSE, R², PHM Score)
- System metrics (GPU usage, RAM)
- Model artifacts (checkpoints)
- Final results and comparison tables

## Report Structure
Standard project reports should include:
1. **Overview**: Project goals and approach
2. **Methodology**: Model architecture details (CNN scales, attention, LSTM paths)
3. **Experiments**: Configuration, training procedures, evaluation metrics
4. **Results**: Performance metrics, generalization analysis, ablation studies
5. **Discussion**: Interpretation, limitations, future work
6. **Reproducibility**: Instructions, environment details, random seeds

## Usage
Use this skill when you need to:
- Generate figures for academic papers or reports
- Create visualizations from notebook data
- Produce experiment tracking reports
- Document results for sharing or publication
- Prepare final project documentation

## Commands
To work with notebooks:
```bash
# Launch Jupyter for EDA
jupyter notebook notebooks/01_EDA.ipynb

# Launch Jupyter for results analysis  
jupyter notebook notebooks/03_results_analysis.ipynb

# Convert notebook to script (if needed)
jupyter nbconvert --to script notebooks/03_results_analysis.ipynb

# Execute notebook programmatically
jupyter nbconvert --to notebook --execute notebooks/03_results_analysis.ipynb --output executed_results.ipynb
```