#!/bin/bash

# Set source and destination data paths
SRC_DATA_PATH="/workspace/budget_forcing_emergence/results_data/"
DST_DATA_PATH="${PWD}/results_data/"

#spec_path=gemma-2-2b-it_AIME_2024_09-13_19-31-41
#SRC_DATA_PATH="/workspace/budget_forcing_emergence/results_data/${spec_path}/"
#DST_DATA_PATH="${PWD}/results_data/${spec_path}"



# Create destination directory if it doesn't exist
mkdir -p "${DST_DATA_PATH}"

# Transfer files from remote to local
rsync -avz --progress vast-gpu:"${SRC_DATA_PATH}" "${DST_DATA_PATH}"
