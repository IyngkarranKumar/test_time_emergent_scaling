#!/bin/bash

# Auto-generated multi-run script
# Generated from multi_run.yaml

# Total combinations: 3

# Experiment 1/3
# dataset_name=math-ai/aime25
# no normalisation for aimes
python3 main.py \
        --num_samples 30 \
        --batch_size 5 \
        --num_completions 5 \
        --model_name Qwen/Qwen2.5-14B-Instruct \
        --quantization 8 \
        --start_token_budget 9 \
        --end_token_budget 14 \
        --SAVE_BOOL \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.6 \
        --dataset_name math-ai/aime25


# Experiment 2/3
# dataset_name=Maxwell-Jia/AIME_2024
#no normalisation for aimes
python3 main.py \
        --num_samples 30 \
        --batch_size 5 \
        --num_completions 5 \
        --model_name Qwen/Qwen2.5-14B-Instruct \
        --quantization 8 \
        --start_token_budget 9 \
        --end_token_budget 14 \
        --SAVE_BOOL \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.6 \
        --dataset_name Maxwell-Jia/AIME_2024


# Experiment 3/3
# dataset_name=Idavidrein/gpqa
python3 main.py \
        --num_samples 30 \
        --batch_size 5 \
        --num_completions 5 \
        --model_name Qwen/Qwen2.5-14B-Instruct \
        --quantization 8 \
        --start_token_budget 9 \
        --end_token_budget 14 \
        --normalise_over_solution_set \
        --solution_set_batch_size 4 \
        --SAVE_BOOL \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.6 \
        --dataset_name Idavidrein/gpqa

