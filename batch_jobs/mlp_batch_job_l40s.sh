#!/bin/bash

# #SLURM setup 
# #SLURM basically puts us on a machine with our specs
#SBATCH --job-name=batch_job
#SBATCH --output=slurm_logs/%j/output.txt
#SBATCH --error=slurm_logs/%j/error.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=PGR-Standard
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40s:2 #others will do too
#SBATCH --nodelist=scotia[01-08]
#SBATCH --time=12:00:00 #default 

#check if script argument provided 
if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_to_run>"
    echo "Example: $0 batch_jobs/run_AIME24.sh"
    exit 1
fi

SCRIPT_TO_RUN="$1"

if [ ! -f "$SCRIPT_TO_RUN" ]; then
    echo "Error: Script $SCRIPT_TO_RUN not found"
    exit 1
fi

echo "🚨🚨 MAKE SURE THERE ARE NO BREAKPOINTS IN CODE 🚨🚨"
echo "Running script: $SCRIPT_TO_RUN"

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

# Dynamically set HF_CACHE_ITEMS based on which model/script is run
DATASET_ITEMS="Maxwell-Jia___aime_2024,math-ai__aime25,Idavidrein___gpqa"
MODEL_ITEMS=""

# Detect which model(s) to include based on script name
case "$SCRIPT_TO_RUN" in
    *phi*|*Phi*)
        MODEL_ITEMS="models--microsoft--Phi-4-reasoning-plus"
        ;;
    *deepseek*|*DeepSeek*)
        MODEL_ITEMS="models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B"
        ;;
    *qwq*|*QwQ*)
        MODEL_ITEMS="models--Qwen--QwQ-32B"
        ;;
    *)
        echo "Warning: No known model keyword found in script name. Defaulting to Microsoft Phi-4."
        MODEL_ITEMS="models--microsoft--Phi-4-reasoning-plus"
        ;;
esac

export HF_CACHE_ITEMS="${MODEL_ITEMS},${DATASET_ITEMS}" # Set before run

bash scripts/setup_compute_node.sh

# Run experiments in background so we can catch signals
bash "$SCRIPT_TO_RUN" &
EXPERIMENT_PID=$!

# Wait for the experiment to complete or for a signal
wait $EXPERIMENT_PID

# If we get here, experiments completed normally
echo "Experiments completed successfully"
bash scripts/transfer_results_to_head.sh