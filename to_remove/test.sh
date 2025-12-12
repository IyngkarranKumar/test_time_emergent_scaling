#!/bin/bash

echo "HELLO WORLD!"echo "hello world"

#rsync -avzch --progress -e "ssh -p 3333 -o StrictHostKeyChecking=no" ~/budget_forcing_emergence/main_results_data root@localhost:/data/budget_forcing_emergence/ -vvv
#rsync -avz -e "ssh -p 13743" root@91.150.160.38:/workspace/budget_forcing_emergence/vast_source_checksums.txt ~/budget_forcing_emergence/
#sha256sum -c vast_source_checksums.txt

# This script will transfer result files from vast.ai instance back to your local machine

# Set the remote source base directory on the vast.ai machine
REMOTE_BASE_DIR="/workspace/budget_forcing_emergence"
vast_gpu_ip=178.164.41.219
vast_gpu_port=41605

# List of result directories to transfer (relative to remote base dir)
spec_paths=(
    main_results_data/DeepSeek-R1-Distill-Qwen-1.5B_AIME_2024
    main_results_data/DeepSeek-R1-Distill-Qwen-7B_AIME_2024
    main_results_data/DeepSeek-R1-Distill-Qwen-14B_AIME_2024
)

for spec_path in "${spec_paths[@]}"; do
    REMOTE_DATA_PATH="${REMOTE_BASE_DIR}/${spec_path}/"
    LOCAL_DATA_PATH="${PWD}/${spec_path}/"

    echo "Transferring from vast-gpu (${vast_gpu_ip}:${vast_gpu_port}): ${REMOTE_DATA_PATH}"
    echo "        to local: ${LOCAL_DATA_PATH}"

    # Create local directory if it doesn't exist
    mkdir -p "${LOCAL_DATA_PATH}"

    # Transfer files from vast-gpu to local via rsync
    rsync -avzhP --checksum --timeout=600 --progress -e "ssh -p ${vast_gpu_port}" "root@${vast_gpu_ip}:${REMOTE_DATA_PATH}" "${LOCAL_DATA_PATH}"
done