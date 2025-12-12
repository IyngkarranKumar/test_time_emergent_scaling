#!/bin/bash

batch_size=5
start_token_budget=7
end_token_budget=10
ATTENTION_TYPE=flash
force_end_batch_size=10
solution_set_batch_size=20
#sample_idxs_range=(0,1)

python3 main.py \
        --batch_size $batch_size \
        --ATTENTION_TYPE $ATTENTION_TYPE \
        --start_token_budget $start_token_budget \
        --end_token_budget $end_token_budget \
        --force_end_batch_size $force_end_batch_size \
        --solution_set_batch_size $solution_set_batch_size \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --dataset_name math-ai/aime25 \
        --num_completions 5 \
        --quantization 8 \
        --normalise_over_solution_set \
        --SAVE_BOOL \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.7 \
        --max_new_tokens_frac 0.0625