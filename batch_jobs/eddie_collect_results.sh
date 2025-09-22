#!/bin/bash
#$ -q gpu
#$ -l h=node3b06.ecdf.ed.ac.uk
#$ -l h_vmem=1G
#$ -cwd
#$ -j y
#$ -o job_output.log

export scratch_disk_dir=/exports/eddie/scratch/s2517451 #set before shell envs set - always good to enforce this again

# Your script to get results
source scripts/setup_shell_environment.sh

source scripts/transfer_results_to_head.sh