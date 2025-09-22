#!/bin/bash

# Auto-generated multi-run script
# Generated from multi_run.yaml

# Total combinations: 3

# Experiment 1/3
# dataset_name=Idavidrein/gpqa
# python3 main.py \
#         --num_samples 30 \
#         --batch_size 2 \
#         --num_completions 5 \
#         --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#         --quantization 8 \
#         --start_token_budget 13 \
#         --end_token_budget 13 \
#         --normalise_over_solution_set \
#         --solution_set_batch_size 4 \
#         --SAVE_BOOL \
#         --inference_engine vllm \
#         --vllm_gpu_memory_utilization 0.6 \
#         --dataset_name Idavidrein/gpqa

# echo "End of GPQA run" 


# Experiment 2/3
# dataset_name=math-ai/aime25
# trying a smaller batch size to prevent cuda lib error. 
python3 main.py \
        --num_samples 4 \
        --batch_size 2 \
        --num_completions 3 \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --quantization 8 \
        --start_token_budget 7 \
        --end_token_budget 8 \
        --normalise_over_solution_set \
        --solution_set_batch_size 10 \
        --SAVE_BOOL \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.6 \
        --dataset_name Idavidrein/gpqa




# Experiment 3/3
# dataset_name=Maxwell-Jia/AIME_2024
# python3 main.py \
#         --num_samples 30 \
#         --batch_size 5 \
#         --num_completions 5 \
#         --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
#         --quantization 8 \
#         --start_token_budget 9 \
#         --end_token_budget 13 \
#         --normalise_over_solution_set \
#         --solution_set_batch_size 20 \
#         --SAVE_BOOL \
#         --inference_engine vllm \
#         --vllm_gpu_memory_utilization 0.6 \
#         --dataset_name Maxwell-Jia/AIME_2024
# echo "End of AIME2024 run" 



