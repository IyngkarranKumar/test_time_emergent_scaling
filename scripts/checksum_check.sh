#!/bin/bash

#see if chcksums match 

login_node_dir="/home/eidf029/eidf029/s2517451-infk8s/budget_forcing_emergence/results_data"
target_dir_pattern="Deepseek-R1-Distill-Qwen-1.5B_gpqa"
echo "Target directory pattern: $target_dir_pattern"

# Store only the hashes of all files recursively within the login node dir
checksums_login_node=$(cd "$login_node_dir" && find . -type f -exec md5sum {} \; | awk '{print $1}' | sort)


echo $checksums_login_node

#for those that don't replace with new one from outputs pvc 