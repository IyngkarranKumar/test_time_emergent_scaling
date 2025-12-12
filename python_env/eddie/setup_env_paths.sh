#!/bin/bash

eddie_scratch_disk="/exports/eddie/scratch/s2517451"

# Remove old conda config
conda config --remove pkgs_dirs /exports/eddie3_homes_local/s2517451/miniconda3/pkgs
conda config --remove envs_dirs /exports/eddie3_homes_local/s2517451/miniconda3/envs

# Add correct conda config
conda config --add pkgs_dirs $eddie_scratch_disk/miniconda/pkgs
conda config --add envs_dirs $eddie_scratch_disk/miniconda/envs

mamba config -add pkgs_dirs $eddie_scratch_disk/miniconda/pkgs
mamba config -add envs_dirs $eddie_scratch_disk/miniconda/envs

echo "export TMPDIR=$eddie_scratch_disk/tmp" >> ~/.bashrc

#tmp dir fixes

echo "Done! Run 'source ~/.bashrc' to apply changes."

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1  
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1