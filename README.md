# Emergent scaling at test-time

Project repository for investigating emergent test-time scaling in language reasoning models. 


## Overview

Code to investigate emergent scaling behaviour at test-time with language reasoning models (LRM). We aim to establish if there are sharp, non-linear increases in LRM performance as test-time compute budget is scaled, in an anologous study to the train-time scaling case ([Wei et al. 2022](https://arxiv.org/pdf/2206.07682)). Both discrete metrics (accuracy) and continuous metrics are tracked to see if non-linear scaling behaviour of discrete metrics can be explained by smoother scaling of the continous metrics. 

#### Research Questions

- Do test-time compute scaling laws exhibit sharp, non-linear (emergent) scaling behaviour, similar to what has been observed when scaling train-time compute([Wei et al. 2022](https://arxiv.org/pdf/2206.07682)) ?
- Is observed emergent scaling behaviour contingent on the scoring metric ([Schaeffer et al. 2023](https://arxiv.org/abs/2304.15004)) or present across discrete **and** continuous metrics?
- What influence do the following properties have on the tendency to observe emergent test-time scaling behaviour?
  - Task distribution
  - Reasoning trajectory
  - Language reasoning model (LRM) used for evaluation
  - Number of LRM model parameters

## Usage

### Environment setup

Set the relevant environment variables within ``env_vars/.env.main`` (``HF_TOKEN``, CUDA variables, etc.)

### Python environment 🐍
 
Run the command ``bash python_env/setup_python_environment.sh`` to set up the environment. This project uses the vLLM library, which requires specific package dependencies and can be fragile at times. If you run into dependency conflicts, try reinstalling the vLLM package, and use the script ``python_env/check_installation.sh`` to check that things are working. 

### Key Components

### Main Execution
- `main.py`: Primary experimental pipeline (lines 300-800 contain the main generation loop)
- `conf.py`: Workflow configuration

### Core Utilities
- `utils`: Contains utility functions for workflow, scoring, and analysis. 

### Analysis Framework
- `utils/analysis_utils.py`: Analysis toolkit
  - `AnalysisWorkflow` class: High-level analysis interface
  - `plot_data()`: Visualization of scaling curves
  - `get_samplewise_breakthrough_and_skew()` (lines 400-500): Emergence metrics
- `analysis.py`: Example analysis workflows

### Experimental Configurations
- `config/`: YAML configuration files for different experimental setups
- `batch_jobs/`: SLURM/GridEngine scripts for cluster execution

## Config



### Key Configuration Parameters
```yaml
model_name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
dataset_name: "Idavidrein/gpqa" 
start_token_budget: 7  # 2^7 = 128 tokens
end_token_budget: 11   # 2^11 = 2048 tokens
num_completions: 3
normalise_over_solution_set: true
```

## Analysis 

The `AnalysisWorkflow` class provides high-level analysis tools:

```python
# Load and analyze results
workflow = utils.AnalysisWorkflow(paths=result_paths, results_save_dir="analysis")

# Generate scaling curves
workflow.aggregate_plots(SHOW=True, SAVE=True)

# Compute emergence scores
emergence_scores = workflow.emergence_scores(model_name="deepseek", dataset_name="gpqa")

# Find highest emergence samples  
top_samples = workflow.get_top_k_max_emergence_samples(k=4, plot=True)
```

## GPU requirements

- **GPU Memory**: 12GB recommended for 7B models, 48GB for 32B models
- **Inference**: Supports both single-GPU and multi-GPU setups



