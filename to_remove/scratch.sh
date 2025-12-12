

path="/home/s2517451/budget_forcing_emergence/results_data/DeepSeek-R1-Distill-Qwen-1.5B_aime25_11-01_16-14-56"
path="/home/s2517451/budget_forcing_emergence/results_data/DeepSeek-R1-Distill-Qwen-7B_aime25_11-01_11-42-17"

# Determine if path is a directory or file on remote
is_dir=$(ssh mlp_head "[ -d \"$path\" ] && echo dir || echo file")


if [ "$is_dir" = "dir" ]; then
    echo "It's a directory"
    base_name=$(basename "$path")
    local_dir="/tmp/$base_name"
    
    echo "Downloading directory from mlp_head..."
    mkdir -p "$local_dir"
    rsync -avzc --progress "mlp_head:$path/" "$local_dir/"

    remote_dir="~/budget_forcing_emergence/transfer_data/$base_name"
    echo "Uploading directory to EIDF..."
    ssh EIDF "mkdir -p \"$remote_dir\""
    rsync -avzc --progress "$local_dir/" "EIDF:$remote_dir"

    rm -rf "$local_dir"
else
    # It's a file
    echo "It's a file"
    filename=$(basename "$path")
    dirpath=$(dirname "$path")
    local_dir="/tmp/$(basename "$dirpath")"
    local_file="$local_dir/$filename"

    echo "Downloading file from mlp_head..."
    mkdir -p "$local_dir"
    rsync -avzc --progress mlp_head:"$path" "$local_file"

    remote_dir="~/budget_forcing_emergence/transfer_data/$(basename "$dirpath")"
    echo "Uploading file to EIDF..."
    ssh EIDF "mkdir -p \"$remote_dir\""
    rsync -avzc --progress "$local_file" "EIDF:$remote_dir/"

    rm -rf "$local_dir"
fi
