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

python3 localisation_main.py --config_file localisation_config/GPQA_localisation.yaml

echo "Experiments completed successfully"
bash scripts/transfer_results_to_head.sh