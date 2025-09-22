#!/bin/bash

if command -v mamba &> /dev/null; then
    eval "$(mamba shell.bash hook)" 2>/dev/null
    mamba activate ml_env
else
    eval "$(conda shell.bash hook)" 2>/dev/null
    conda activate ml_env
fi

echo "In run.sh file:"
echo "DEBUG: Using Python: $(which python)"
echo "DEBUG: Environment: $CONDA_DEFAULT_ENV"


# 128 - 2048
# python3 main.py \
#         --num_samples 30 \
#         --batch_size 5 \
#         --num_completions 5 \
#         --model_name Qwen/QwQ-32B \
#         --quantization 8 \
#         --start_token_budget 7 \
#         --end_token_budget 11 \
#         --normalise_over_solution_set \
#         --solution_set_batch_size 20 \
#         --SAVE_BOOL \
#         --save_every_n_mins 175 \
#         --inference_engine vllm \
#         --vllm_gpu_memory_utilization 0.7 \
#         --max_new_tokens_frac 0.0625 \
#         --dataset_name Idavidrein/gpqa

# 128 - 2048
python3 main.py \
        --num_samples 30 \
        --batch_size 5 \
        --num_completions 5 \
        --model_name microsoft/Phi-4-reasoning-plus \
        --quantization 8 \
        --start_token_budget 11 \
        --end_token_budget 11 \
        --normalise_over_solution_set \
        --solution_set_batch_size 20 \
        --SAVE_BOOL \
        --save_every_n_mins 175 \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.7 \
        --max_new_tokens_frac 0.0625 \
        --dataset_name Idavidrein/gpqa


# 128 - 2048
# python3 main.py \
#         --num_samples 30 \
#         --batch_size 5 \
#         --num_completions 5 \
#         --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
#         --quantization 8 \
#         --start_token_budget 7 \
#         --end_token_budget 11 \
#         --normalise_over_solution_set \
#         --solution_set_batch_size 20 \
#         --SAVE_BOOL \
#         --save_every_n_mins 175 \
#         --inference_engine vllm \
#         --vllm_gpu_memory_utilization 0.7 \
#         --max_new_tokens_frac 0.0625 \
#         --dataset_name Idavidrein/gpqa