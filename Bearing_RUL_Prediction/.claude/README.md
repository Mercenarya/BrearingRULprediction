# .claude Directory for Bearing RUL Prediction Project

This directory contains project-specific configurations, rules, and skills for Claude Code when working on the bearing RUL prediction project.

## Contents

### CLAUDE.md
Project-specific guidelines and instructions for Claude Code, including:
- Project overview and technical details
- ML Ops priorities from project documentation
- Standard workflow commands
- Key file structure locations

### Skills/
Project-specific skills that can be invoked with `/skill` or the Skill tool:

1. **mlops-workflow.md** - Standard MLOps workflow (data validation → versioning → smoke test → tracking → cross-dataset eval)
2. **data-ops.md** - Data handling operations (validation, versioning, sources, preprocessing)
3. **training-eval.md** - Model training and evaluation (architecture, configuration, training, evaluation, ablation studies)
4. **report-generation.md** - Report and visualization generation (paper figures, results analysis, experiment tracking)
5. **few-prompts.md** - Reducing permission prompts for common operations

## Usage

### Invoke a skill
```
/skill data-ops
```
/skill training-eval
```

### View project guidelines
Claude Code will automatically reference CLAUDE.md when working in this project directory.

### Reduce permission prompts
Use the few-prompts skill to get recommendations for minimizing interruptions during development workflows.

## Project Context
This configuration supports the bearing RUL prediction project which uses:
- Multi-scale CNN + Cross-modal Attention + Dual-path LSTM architecture
- Stage-aware PHM loss function
- PRONOSTIA and XJTU-SY bearing datasets
- MLOps focus on data integrity, reproducibility, and cross-dataset generalization