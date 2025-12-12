#!/bin/bash

batch_size=5
start_token_budget=11
end_token_budget=13
force_end_batch_size=10
solution_set_batch_size=20
ATTENTION_TYPE=flash
dataset_name=Idavidrein/gpqa

target_run="1.5B"

if [ "$target_run" = "14B" ] || [ "$target_run" = "ALL" ]; then
    python3 main.py \
        --batch_size $batch_size \
        --ATTENTION_TYPE $ATTENTION_TYPE \
        --start_token_budget $start_token_budget \
        --end_token_budget $end_token_budget \
        --force_end_batch_size $force_end_batch_size \
        --solution_set_batch_size $solution_set_batch_size \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --dataset_name $dataset_name \
        --num_completions 5 \
        --quantization 8 \
        --normalise_over_solution_set \
        --solution_set_batch_size 20 \
        --SAVE_BOOL \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.7 \
        --max_new_tokens_frac 0.0625
fi

if [ "$target_run" = "7B" ] || [ "$target_run" = "ALL" ]; then
    python3 main.py \
        --sample_idxs_range "(100,198)" \
        --batch_size $batch_size \
        --ATTENTION_TYPE $ATTENTION_TYPE \
        --start_token_budget 13 \
        --end_token_budget 13 \
        --force_end_batch_size $force_end_batch_size \
        --solution_set_batch_size $solution_set_batch_size \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --dataset_name $dataset_name \
        --num_completions 5 \
        --quantization 8 \
        --normalise_over_solution_set \
        --solution_set_batch_size 20 \
        --SAVE_BOOL \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.7 \
        --max_new_tokens_frac 0.0625
fi

if [ "$target_run" = "1.5B" ] || [ "$target_run" = "ALL" ]; then
    python3 main.py \
        --sample_idxs_range "(110,198)" \
        --batch_size $batch_size \
        --ATTENTION_TYPE $ATTENTION_TYPE \
        --start_token_budget 13 \
        --end_token_budget 13 \
        --force_end_batch_size $force_end_batch_size \
        --solution_set_batch_size $solution_set_batch_size \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --dataset_name $dataset_name \
        --num_completions 5 \
        --quantization 8 \
        --normalise_over_solution_set \
        --solution_set_batch_size 20 \
        --SAVE_BOOL \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.7 \
        --max_new_tokens_frac 0.0625
fi
