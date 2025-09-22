#!/bin/bash

#SBATCH --job-name=transfer_results
#SBATCH --output=slurm_logs/%j_transfer.out
#SBATCH --error=slurm_logs/%j_transfer.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=PGR-Standard-Noble
#SBATCH --nodelist=scotia01
#SBATCH --mem=4G
#SBATCH --time=1:00:00


source ~/.bashrc

source scripts/setup_shell_environment.sh

echo "Transferring results from ${HF_CACHE_PATH} to head node"
bash scripts/transfer_results_to_head.sh

