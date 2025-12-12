#!/bin/bash

#!/bin/bash

# This script will transfer result files from your local machine to a vast.ai instance
# referenced as 'vast-gpu' in your SSH config.

##for checksum 
#rsync -avz -e "ssh -p 54561" /tmp/source_checksums.txt root@171.101.231.99:/workspace/budget_forcing_emergence/
#sha256sum -c /tmp/source_checksums.txt

# Set the remote destination base directory on the vast.ai machine.
REMOTE_BASE_DIR="/workspace/budget_forcing_emergence"
vast_gpu_ip=116.109.111.188
vast_gpu_port=23888

# List of result directories to transfer (relative to local ./results_data/)
spec_paths=(
    main_results_data
)

for spec_path in "${spec_paths[@]}"; do
    #LOCAL_DATA_PATH="${PWD}/results_data/${spec_path}/"
    LOCAL_DATA_PATH="${PWD}/${spec_path}/"
    REMOTE_DATA_PATH="${REMOTE_BASE_DIR}/${spec_path}/"

    echo "Transferring from local: ${LOCAL_DATA_PATH}"
    echo "        to vast-gpu (${vast_gpu_ip}:${vast_gpu_port}): ${REMOTE_DATA_PATH}"

    # Create remote directory if it doesn't exist
    ssh -p "${vast_gpu_port}" "root@${vast_gpu_ip}" "mkdir -p '${REMOTE_DATA_PATH}'"

    # Transfer only files named config.pkl from local to vast-gpu via rsync with specified IP and port
    rsync -avzhP --checksum --timeout=600 --progress -e "ssh -p ${vast_gpu_port}" "${LOCAL_DATA_PATH}" "root@${vast_gpu_ip}:${REMOTE_DATA_PATH}"
    # rsync -avzhP --checksum --timeout=600 --progress \
    #     --include='*/' \
    #     --include='*config*' \
    #     --exclude='*' \
    #     -e "ssh -p ${vast_gpu_port}" \
    #     "${LOCAL_DATA_PATH}" "root@${vast_gpu_ip}:${REMOTE_DATA_PATH}"
done