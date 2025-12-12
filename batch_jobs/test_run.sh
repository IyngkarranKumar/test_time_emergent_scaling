#!/bin/bash

python3 main.py \
        --batch_size 5 \
        --num_completions 5 \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --quantization 8 \
        --start_token_budget 7 \
        --end_token_budget 7 \
        --normalise_over_solution_set \
        --SAVE_BOOL \
        --DISABLE_FLASH_ATTENTION \
        --save_every_n_mins 175 \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.7 \
        --max_new_tokens_frac 0.0625 \
        --dataset_name Idavidrein/gpqa