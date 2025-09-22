#!/bin/bash

# #SLURM setup 
# #SLURM basically puts us on a machine with our specs
#SBATCH --job-name=batch_job
#SBATCH --output=slurm_logs/%j/output.txt
#SBATCH --error=slurm_logs/%j/error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=PGR-Standard,PGR-Standard-Noble,Teach-Standard
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:2 #others will do too
#SBATCH --exclude=scotia08
#SBATCH --time=24:00:00


echo "🚨🚨 MAKE SURE THERE ARE NO BREAKPOINTS IN CODE 🚨🚨"

# Function to copy results back when job is about to be killed (Claude)
cleanup() {
    echo "Job termination signal received, copying results back..."
    bash scripts/transfer_results_to_head.sh
    echo "Emergency backup completed"
    exit 0
}

# Set up trap to catch SIGTERM (sent by SLURM before killing job)
trap cleanup SIGTERM SIGINT

# Your existing setup
source ~/.bashrc
echo "Job running on ${SLURM_JOB_NODELIST}"
mamba activate ml_env
source scripts/setup_shell_environment.sh

#setup compute node 
bash scripts/setup_compute_node.sh

# Run experiments in background so we can catch signals
#bash scripts/run_experiments_7B.sh &
#EXPERIMENT_PID=$!

# Wait for the experiment to complete or for a signal
#wait $EXPERIMENT_PID





#TEST RUN FOR force continues

python3 main.py \
        --num_samples 12 \
        --batch_size 4 \
        --num_completions 1 \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --quantization 8 \
        --start_token_budget 12 \
        --end_token_budget 13 \
        --SAVE_BOOL \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.6 \
        --dataset_name Idavidrein/gpqa



python3 main.py \
        --num_samples 12 \
        --batch_size 4 \
        --num_completions 1 \
        --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --quantization 8 \
        --start_token_budget 12 \
        --end_token_budget 13 \
        --SAVE_BOOL \
        --inference_engine vllm \
        --vllm_gpu_memory_utilization 0.6 \
        --dataset_name math-ai/aime25


#TEST RUN FOR SYSTEM PROMPT

# If we get here, experiments completed normally
echo "Experiments completed successfully"
bash scripts/transfer_results_to_head.sh

