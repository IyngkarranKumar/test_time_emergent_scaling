#!/bin/bash

#SBATCH --job-name=transfer_results
#SBATCH --output=transfer_logs/%j_transfer.out
#SBATCH --error=transfer_logs/%j_transfer.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=PGR-Standard,PGR-Standard-Noble,Teach-Standard

#SBATCH --mem=4G
#SBATCH --time=1:00:00