#!/bin/bash

export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_DEBUG=1
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export CUDA_MEMCHECK=1
export CUDA_SYNC_MEMOPS=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"

python3 main.py \
        --num_samples 30 \
        --batch_size 5 \
        --num_completions 5 \
        --model_name microsoft/Phi-4-reasoning-plus \
        --start_token_budget 11 \
        --end_token_budget 11 \
        --quantization 8 \
        --force_end_batch_size 25 \
        --normalise_over_solution_set \
        --solution_set_batch_size 100 \
        --SAVE_BOOL \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.7 \
        --max_new_tokens_frac 0.0625 \
        --dataset_name math-ai/aime25 2>&1 | tee console.log