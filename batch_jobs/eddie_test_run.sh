
#!/bin/bash

# GridEngine setup
#$ -N test_7b                           # Job name
#$ -o gridengine_logs/$JOB_ID/output.txt      # Output file
#$ -e gridengine_logs/$JOB_ID/error.txt       # Error file
#$ -pe sharedmem 1                       # Parallel environment (1 task)
#$ -q gpu                                # GPU queue
#$ -l h_vmem=32G                         # Reduced memory request
#$ -l gpu=1                              # Request only 1 GPU
#$ -l h_rt=2:00:00                       # Much shorter runtime for test
#$ -cwd                                  # Run in current working directory

source ~/.bashrc
source scripts/setup_shell_environment
mamba activate ml_env 
# Verify environment activation
echo "Current environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Testing torch import..."
python -c "import torch; print(f'Torch version: {torch.__version__}'); print('Torch import successful!')" || {
    echo "ERROR: Torch import failed!"
    echo "Available packages:"
    pip list | grep -i torch
    exit 1
}

export scratch_disk_dir=/exports/eddie/scratch/s2517451 #set before shell envs set - always good to enforce this again
echo "scratch_disk_dir: $scratch_disk_dir"
echo "HF_CACHE_PATH: $HF_CACHE_PATH"



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


echo "Experiments completed successfully"
bash scripts/transfer_results_to_head.sh

