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
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=24:00:00

python3 main.py --config_file config/32B_check.yaml