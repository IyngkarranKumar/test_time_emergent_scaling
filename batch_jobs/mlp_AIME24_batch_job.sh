#!/bin/bash

# #SLURM setup 
# #SLURM basically puts us on a machine with our specs
#SBATCH --job-name=batch_job
#SBATCH --output=slurm_logs/%j/output.txt
#SBATCH --error=slurm_logs/%j/error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=PGR-Standard,Teach-Standard
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:4 #others will do too
#SBATCH --exclude=scotia08
#SBATCH --time=48:00:00


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
export scratch_disk_dir=/disk/scratch/s2517451 #set before shell envs set
source scripts/setup_shell_environment.sh
mamba activate ml_env

echo "scratch_disk_dir: $scratch_disk_dir"
echo "HF_CACHE_PATH: $HF_CACHE_PATH"





#setup compute node 
export HF_CACHE_ITEMS=models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B,models--Qwen--QwQ-32B,models--microsoft--Phi-4-reasoning-plus,Maxwell-Jia___aime_2024 #set before run

bash scripts/setup_compute_node.sh

# Run experiments in background so we can catch signals
bash batch_jobs/run_AIME24.sh &
EXPERIMENT_PID=$!

# Wait for the experiment to complete or for a signal
wait $EXPERIMENT_PID


# If we get here, experiments completed normally
echo "Experiments completed successfully"
bash scripts/transfer_results_to_head.sh

