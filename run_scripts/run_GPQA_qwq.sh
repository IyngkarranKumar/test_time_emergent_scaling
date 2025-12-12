#!/bin/bash

batch_size=5
force_end_batch_size=5
solution_set_batch_size=10
start_token_budget=13
end_token_budget=13
ATTENTION_TYPE=mem_efficient
sample_idxs_range="(125,128)" #think this avoid long sequenvce 127 (but)

python3 main.py \
        --batch_size $batch_size \
        --sample_idxs_range $sample_idxs_range \
        --ATTENTION_TYPE $ATTENTION_TYPE \
        --start_token_budget $start_token_budget \
        --end_token_budget $end_token_budget \
        --force_end_batch_size $force_end_batch_size \
        --solution_set_batch_size $solution_set_batch_size \
        --model_name Qwen/QwQ-32B \
        --dataset_name Idavidrein/gpqa \
        --num_completions 5 \
        --quantization 8 \
        --normalise_over_solution_set \
        --SAVE_BOOL \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.8 \
        --max_new_tokens_frac 0.0625