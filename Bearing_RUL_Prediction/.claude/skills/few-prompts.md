# Few Permission Prompts Skill for Bearing RUL Prediction

This skill reduces permission prompts for common operations in the bearing RUL prediction project.

## Common Operations That Trigger Prompts
During typical workflows, these operations often trigger permission prompts:
- File system operations (reading/writing data files)
- Python script execution (training, evaluation, preprocessing)
- Git operations (if version controlling experiments)
- Pip installations (for dependency management)
- Directory creation/deletion

## Recommended Allowlist Additions
To reduce prompts, consider adding these to your .claude/settings.json:

### Data Operations
- Read/Write access to `data/processed/` directory
- Read access to `src/` directory for executing scripts
- Read access to `notebooks/` directory
- Read access to `config.yaml`
- Read/Write access to `checkpoints/` directory
- Read/Write access to `results/` directory (if created)
- Read access to `requirements.txt`

### Python Execution
- Allow execution of Python scripts in src/ directory:
  - `src/data/validate_data.py`
  - `src/data/generate_registry.py` 
  - `src/training/train.py`
  - `src/evaluation/evaluate.py`
  - `src/evaluation/cross_dataset_evaluate.py`
  - `src/evaluation/ablation.py`
  - `src/data/download.py`
  - `src/data/preprocessing.py`
  - `src/data/sliding_window.py`

### Notebook Operations
- Allow Jupyter notebook execution in notebooks/ directory
- Allow reading/writing .ipynb files

## Permission Mode Strategies
For this project, consider these approaches:

### 1. Session-Specific Allowlist
Add project-specific permissions to .claude/settings.local.json:
```json
{
  "permissions": {
    "allow": [
      "Python(src\\data\\validate_data.py)",
      "Python(src\\data\\generate_registry.py)", 
      "Python(src\\training\\train.py)",
      "Python(src\\evaluation\\evaluate.py)",
      "Python(src\\evaluation\\cross_dataset_evaluate.py)",
      "Read(data\\processed\\*)",
      "Write(data\\processed\\registry.csv)",
      "Read(checkpoints\\*)",
      "Write(checkpoints\\*)"
    ]
  }
}
```

### 2. Broad Allowlist for Development
During active development, consider:
- Allowing read/write to entire project directory for file operations
- Allowing Python execution for all scripts in src/
- Temporarily relaxing prompts while iterating

## Usage
Use this skill when:
- Setting up a new development environment
- Experiencing frequent permission interrupts during workflows
- Preparing for demonstrations or presentations
- Setting up CI/CD or automated pipelines

To check current permissions, review:
- .claude/settings.local.json (session-specific)
- .claude/settings.json (user/global)
- Global Claude Code settings