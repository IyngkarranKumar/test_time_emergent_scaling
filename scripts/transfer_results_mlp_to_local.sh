#!/bin/bash

# Set source and destination data paths
SRC_DATA_PATH="~/budget_forcing_emergence/results_data/"
DST_DATA_PATH="${PWD}/results_data"

spec_path=DeepSeek-R1-Distill-Qwen-14B_gpqa_09-22_03-31-56
SRC_DATA_PATH="~/budget_forcing_emergence/results_data/${spec_path}/"
DST_DATA_PATH="${PWD}/results_data/${spec_path}"



# Create destination directory if it doesn't exist
mkdir -p "${DST_DATA_PATH}"

# Transfer files from remote to local
rsync -avz --progress mlp_head:"${SRC_DATA_PATH}" "${DST_DATA_PATH}"
