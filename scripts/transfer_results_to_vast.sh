#!/bin/bash

# This script will transfer result files from your local machine to a vast.ai instance
# referenced as 'vast-gpu' in your SSH config.

# Set the remote destination base directory on the vast.ai machine.
REMOTE_BASE_DIR="/workspace/budget_forcing_emergence/results_data"

# List of result directories to transfer (relative to local ./results_data/)
spec_paths=(
    debug/DeepSeek-R1-Distill-Qwen-14B_gpqa_10-20_04-37-49
)

for spec_path in "${spec_paths[@]}"; do
    LOCAL_DATA_PATH="${PWD}/results_data/${spec_path}/"
    REMOTE_DATA_PATH="${REMOTE_BASE_DIR}/${spec_path}/"

    echo "Transferring from local: ${LOCAL_DATA_PATH}"
    echo "           to vast-gpu: ${REMOTE_DATA_PATH}"

    # Create remote directory if it doesn't exist
    ssh vast-gpu "mkdir -p '${REMOTE_DATA_PATH}'"

    # Transfer files from local to vast-gpu via rsync
    rsync -avzc --timeout=600 --progress "${LOCAL_DATA_PATH}" "vast-gpu:${REMOTE_DATA_PATH}"
done

