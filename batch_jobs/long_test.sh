#!/bin/bash

#to test long sequence generation
python3 main.py \
        --num_samples 5 \
        --batch_size 5 \
        --num_completions 5 \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --quantization 8 \
        --start_token_budget 13 \
        --end_token_budget 14 \
        --normalise_over_solution_set \
        --SAVE_BOOL \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.7 \
        --max_new_tokens_frac 0.0625 \
        --dataset_name math-ai/aime25 \
        --ATTENTION_TYPE mem_efficient \