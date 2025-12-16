# Emergent abilities in language models at test-time

Project repository for investigating emergent test-time scaling in language reasoning models. 

## Overview

We investigate if there are sharp, non-linear increases in LRM performance as test-time compute budget is scaled, in an analogous study to the train-time scaling case ([Wei et al. 2022](https://arxiv.org/pdf/2206.07682)). Discrete metrics (accuracy) and continuous metrics are tracked to see if abrupt scaling of discrete metrics can be explained by smoother scaling of the continous metrics. 

#### Research Questions

1.  To what extent does emergent scaling vary across different datasets? Are the returns to test-time compute best modelled by a log-linear relationship, or are there cases where other fits (such as piecewise linear) are more appropriate?
2.  To what extent do individual dataset instances consistently exhibit emergent test-time scaling? Across multiple models, do the same instances consistently exhibit sharp, abrupt increases in performance?
3.  To what extent does emergent scaling vary across different model families? Do some reasoning models display sharper scaling behaviour than others, or is this behaviour consistent across model families?
4.  To what extent does emergent scaling vary across model size (number of parameters)? Do larger models exhibit sharper scaling than smaller ones, or does model size have a negligible influence on emergent test-time scaling?
5.  Is the presence of emergent test-time scaling contingent on the metric used
for evaluation? Does emergent scaling only occur when using discrete evaluation metrics, or is it present when tracking continuous evaluation metrics too?

## Using this repository

Begin with: 

```bash 
git clone https://github.com/IyngkarranKumar/test_time_emergent_scaling.git
```

### Environment setup

Set the relevant environment variables within ``env_vars/.env.main`` (``HF_TOKEN``, CUDA variables, etc.)

### Python environment 🐍

This project uses [mamba](https://mamba.readthedocs.io/en/latest/) for python package management - make sure it is installed on your system. 

Run the command ``bash python_env/setup_python_environment.sh`` to set up the environment. This project uses the vLLM library, which requires specific package dependencies and can be fragile. If you run into dependency conflicts, try reinstalling the vLLM package, and use the script ``python_env/check_installation.sh`` to check that things are working (you should see: "✅ vLLM engine loaded successfully").

### Generating scaling data

Text generation and scoring can be done via command line:

```bash
python3 main.py \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --dataset_name Maxwell-Jia/AIME_2024 \
    --start_token_budget 7 \
    --end_token_budget 10 \
    --batch_size 5 \
    --SAVE_BOOL \
```

Or via yaml config files if working in an IDE (recommended): 

```
python3 main.py --config_file config/example.yaml
```

### Analysis

The `AnalysisWorkflow` class provides high-level analysis tools for the generated responses:

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

### Config parameters

The main parameters to modify for each experiment are:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `model_name` | str | HuggingFace model identifier | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| `dataset_name` | str | Dataset identifier (see supported datasets below) | `math-ai/aime25` |
| `start_token_budget` | int | Starting token budget (log2 scale) | `7` (= 128 tokens) |
| `end_token_budget` | int | Ending token budget (log2 scale) | `11` (= 2048 tokens) |
| `batch_size` | int | Number of samples processed per batch | `4` |
| `SAVE_BOOL` | bool | Enable/disable saving results (in directory: ``results_data``) | `True` |

#### Supported Datasets:  
- `math-ai/aime25`
- `Idavidrein/gpqa`
- `Maxwell-Jia/AIME_2024`
- `ProCreations/SimpleMath` (debug)
- `openai/gsm8k` (debug)

Additional config parameters are given at the very end of this page. 

## Directory structure

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

## GPU requirements

- **GPUs**: By default, two GPUs are required - one for the vLLM engine (which does most of the text generation) and one for the HuggingFace model (which is needed to calculate probabilities of each candidate solution). The workflow should work for a single GPU setup, but is better optimized for two GPUs. 



<hr>
<hr>
<br>


## Additional configuration parameters

### Number of samples and generations
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_samples` | int/None | `None` | Number of samples to process (None = full dataset) |
| `sample_idxs_range` | tuple/None | `None` | Range of sample indices `[start, end)` |
| `num_completions` | int | `1` | Number of generations per sample |

**Note:** Cannot set both `num_samples` and `sample_idxs_range` simultaneously.

### Batch Processing
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | `4` | Primary batch size for processing |
| `force_end_batch_size` | int/None | `None` | Batch size for forced ending (defaults to `batch_size * num_completions`) |
| `solution_set_batch_size` | int | `50` | Batch size for solution set evaluation (auto-set to `4` for GPQA) |

### Model Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quantization` | int/None | `None` | Model weight precision (e.g., `8` for 8-bit, `None` for fp16) |
| `ATTENTION_TYPE` | str | `"mem_efficient"` | Attention mechanism: `"flash"`, `"mem_efficient"`, or `"vanilla"` |


### Inference Engine (vLLM settings)
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inference_engine` | str | `"vllm"` | Engine type: `"hf"` or `"vllm"` (VLLM requires CUDA) |
| `vllm_gpu_memory_utilization` | float | `0.4` | GPU memory fraction for VLLM (0.0-1.0) |
| `vllm_max_num_seqs` | int | `64` | Maximum number of sequences for VLLM |
| `max_model_len` | int/None | `None` | Max context length (auto-computed from token budget + buffer) |

### Generation Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_new_tokens_frac` | float | `0.125` | Number of tokens to generate per-step, measured by fraction of token budget (e.g: if generating response of 128 tokens, this will generate 0.125*128 = 16 tokens at a time) |
| `final_answer_budget` | int | `10` | Reserved tokens for final answer |
| `force_continue` | str | `" Hmm, but let me keep thinking... <think>"` | Continuation prompt |

### Scoring Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalise_over_solution_set` | bool | `True` | Normalize scores across solution set |

### Save Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SAVE_BOOL` | bool | `False` | Enable result saving |
| `TIME_BASED_SAVING` | bool | `True` | Save at time intervals (vs. per token budget) |
| `save_every_n_mins` | int | `120` | Save interval in minutes (when `TIME_BASED_SAVING=True`) |
