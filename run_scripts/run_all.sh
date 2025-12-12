#!/bin/bash

MODELS=(
    "google/gemma-2-2b-it"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)
DATASETS=(
    "Idavidrein/gpqa"
    "math-ai/aime25"
    "Maxwell-Jia/AIME_2024"
)
start_token_budget=7
end_token_budget=10
max_new_tokens_frac=0.125
batch_size=4
vllm_gpu_memory_utilization=0.8
save_ever_n_mins=60
num_samples=12
num_completions=3

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        start_time=$(date '+%Y-%m-%d %H:%M:%S')
        echo "==============================================="
        echo "Starting run at $start_time"
        echo "Model:    $model"
        echo "Dataset:  $dataset"
        echo "==============================================="
        python3 main.py \
            --num_samples $num_samples \
            --batch_size $batch_size \
            --num_completions 5 \
            --model_name "$model" \
            --quantization 8 \
            --start_token_budget $start_token_budget \
            --end_token_budget $end_token_budget \
            --normalise_over_solution_set \
            --solution_set_batch_size 4 \
            --SAVE_BOOL \
            --save_ever_n_mins $save_ever_n_mins \
            --inference_engine vllm \
            --vllm_gpu_memory_utilization $vllm_gpu_memory_utilization \
            --dataset_name "$dataset"
    done
done
