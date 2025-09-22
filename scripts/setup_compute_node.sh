#!/bin/bash


# Print environment info

echo "USER: $USER"
echo "HOSTNAME: $HOSTNAME"
echo "SCRATCH_DISK_DIR: $scratch_disk_dir"

echo "Creating data directory in scratch if it doesn't exist"
mkdir -p $scratch_disk_dir/data

# Create HF cache directory in scratch if it doesn't exist
echo "Creating HF cache directory in ${scratch_disk_dir} if it doesn't exist"
mkdir -p $scratch_disk_dir/hf_cache


# Copy HF cache from home to scratch disk
#we ignore really large directories like s1.1-32B
echo "Copying selected HF cache from $HEAD_NODE_HF_CACHE_PATH to $HF_CACHE_PATH"
echo "You can specify models/datasets to copy via the HF_CACHE_ITEMS environment variable (comma-separated, e.g. 'models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B,Idavidrein___gpqa,math-ai___aime25')"
IFS=',' read -ra HF_CACHE_ITEMS <<< "${HF_CACHE_ITEMS:-}"

if [ ${#HF_CACHE_ITEMS[@]} -eq 0 ]; then
    echo "No specific HF cache items specified, copying all except *32B*..."
    rsync -ah --info=progress2 --exclude='*32B*' $HOME/hf_cache/* $scratch_disk_dir/hf_cache/
else
    echo "Copying only specified HF cache items: ${HF_CACHE_ITEMS[*]}"
    for item in "${HF_CACHE_ITEMS[@]}"; do
        echo "Copying $item ..."
        rsync -ah --info=progress2 "$HOME/hf_cache/$item" $scratch_disk_dir/hf_cache/
    done
fi

echo "✓ HF cache copy complete"
