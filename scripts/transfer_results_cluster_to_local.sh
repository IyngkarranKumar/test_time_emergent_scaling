#!/bin/bash

cluster=EIDF

spec_paths=(
    DeepSeek-R1-Distill-Qwen-14B_gpqa_10-20_04-37-49
)

for spec_path in "${spec_paths[@]}"; do
    if [[ "${cluster}" == "mlp_head" || "${cluster}" == "eddie_ecdf" ]]; then
        SRC_DATA_PATH="~/budget_forcing_emergence/results_data/${spec_path}/"
    elif [[ "${cluster}" == "EIDF" ]]; then
        SRC_DATA_PATH="/home/eidf029/eidf029/s2517451-infk8s/budget_forcing_emergence/results_data/${spec_path}/"
    else
        echo "Unknown cluster: ${cluster}"
        exit 1
    fi
    DST_DATA_PATH="${PWD}/results_data/debug/${spec_path}"

    # Create destination directory if it doesn't exist
    mkdir -p "${DST_DATA_PATH}"

    # Transfer files from remote to local
    rsync -avzc --timeout=600 --progress ${cluster}:"${SRC_DATA_PATH}" "${DST_DATA_PATH}"
done

