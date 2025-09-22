#!/bin/bash

echo "Setting up environment variables with env/.env.mlp_compute"
set -a
source env/.env.mlp_compute
set +a

# Print environment info

echo "USER: $USER"
echo "HOSTNAME: $HOSTNAME"
echo "SCRATCH_DISK_DIR: $scratch_disk_dir"

echo "Creating data directory in scratch if it doesn't exist"
mkdir -p $scratch_disk_dir/data

# Create HF cache directory in scratch if it doesn't exist
echo "Creating HF cache directory in scratch if it doesn't exist"
mkdir -p $scratch_disk_dir/hf_cache


# Copy HF cache from home to scratch disk
#we ignore really large directories like s1.1-32B
echo "Copying HF cache from $HEAD_NODE_HF_CACHE_PATH to $HF_CACHE_PATH"
echo "Starting copy of HF cache..."
rsync -ah --info=progress2 \
    --exclude='s1.1-32B*' \
    --exclude='*Qwen2.5-14B*' \
    --exclude='*Qwen2.5-7B*' \
    --exclude='*7B*' \
    --exclude='*8B*' \
    --exclude='*9B*' \
    --exclude='*1[0-9]B*' \
    --exclude='*2[0-9]B*' \
    --exclude='*3[0-9]B*' \
    --exclude='*[4-9][0-9]B*' \
    --exclude='*[0-9][0-9][0-9]B*' \
    $HOME/hf_cache/* $scratch_disk_dir/hf_cache/


echo "✓ HF cache copy complete"
