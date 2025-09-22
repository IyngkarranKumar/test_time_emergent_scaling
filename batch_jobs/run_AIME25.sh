#!/bin/bash




# 256 - 2048
python3 main.py \
        --num_samples 30 \
        --batch_size 5 \
        --num_completions 5 \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --quantization 8 \
        --start_token_budget 8 \
        --end_token_budget 11 \
        --normalise_over_solution_set \
        --solution_set_batch_size 20 \
        --SAVE_BOOL \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.7 \
        --max_new_tokens_frac 0.0625 \
        --dataset_name math-ai/aime25 \

# 256 - 2048
python3 main.py \
        --num_samples 30 \
        --batch_size 5 \
        --num_completions 5 \
        --model_name Qwen/QwQ-32B \
        --quantization 8 \
        --start_token_budget 8 \
        --end_token_budget 11 \
        --normalise_over_solution_set \
        --solution_set_batch_size 20 \
        --SAVE_BOOL \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.7 \
        --max_new_tokens_frac 0.0625 \
        --dataset_name math-ai/aime25 \

# 256 - 2048
python3 main.py \
        --num_samples 30 \
        --batch_size 5 \
        --num_completions 5 \
        --model_name microsoft/Phi-4-reasoning \
        --quantization 8 \
        --start_token_budget 8 \
        --end_token_budget 11 \
        --normalise_over_solution_set \
        --solution_set_batch_size 20 \
        --SAVE_BOOL \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.7 \
        --max_new_tokens_frac 0.0625 \
        --dataset_name math-ai/aime25
