#!/bin/bash

# GridEngine setup
#$ -N batch_job                           # Job name
#$ -o gridengine_logs/$JOB_ID/output.txt      # Output file
#$ -e gridengine_logs/$JOB_ID/error.txt       # Error file
#$ -q gpu                                # GPU queue
#$ -l gpu=2                            # Request 4 GPUs (A100s)
#$ -l h_rt=24:00:00                      # Runtime limit
#$ -cwd                                  # Run in current working directory
#$ -notify                               # CRITICAL: Enable notification before kill

#check if script argument provided 
if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_to_run>"
    echo "Example: $0 batch_jobs/run_GPQA.sh"
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
trap cleanup SIGUSR1 SIGUSR2 SIGTERM SIGINT

echo 
source scripts/setup_shell_environment.sh
source scripts/cluster_scripts.sh


# Your existing setup
source ~/.bashrc
echo "Job running on ${SLURM_JOB_NODELIST}"

#environment activation 
# Try both mamba and conda hooks
if command -v mamba &> /dev/null; then
    eval "$(mamba shell.bash hook)" 2>/dev/null
    echo "Using mamba"
    mamba activate ml_env
else
    eval "$(conda shell.bash hook)" 2>/dev/null  
    echo "Using conda"
    conda activate ml_env
fi

# Verify environment activation
echo "Current environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Testing torch import..."

# Apply mapping between GPU UUIDS and integer indices
echo "Current CUDA DEVICES: $CUDA_VISIBLE_DEVICES"
if [[ "$CUDA_VISIBLE_DEVICES" == *"GPU-"* ]]; then
    MAPPED_INDICES=$(get_gpu_indices)
    if [[ -n "$MAPPED_INDICES" ]]; then
        export CUDA_VISIBLE_DEVICES="$MAPPED_INDICES"
        echo "Mapped assigned GPUs to indices: $CUDA_VISIBLE_DEVICES"
    else
        echo "Warning: Could not map UUIDs, keeping original assignment"
    fi
fi
echo "CUDA DEVICES after remapping: $CUDA_VISIBLE_DEVICES"

echo "Shared object library path: $LD_LIBRARY_PATH"

export scratch_disk_dir=/exports/eddie/scratch/s2517451 #set before shell envs set - always good to enforce this again

echo "scratch_disk_dir: $scratch_disk_dir"
echo "HF_CACHE_PATH: $HF_CACHE_PATH"


# Run experiments in background so we can catch signals
bash "$SCRIPT_TO_RUN" &
EXPERIMENT_PID=$!

# Wait for the experiment to complete or for a signal
wait $EXPERIMENT_PID


# If we get here, experiments completed normally
echo "Experiments completed successfully"
bash scripts/transfer_results_to_head.sh

