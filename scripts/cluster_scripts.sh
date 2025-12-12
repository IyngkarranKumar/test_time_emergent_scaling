# Simplified version that handles both GPU and MIG devices
get_gpu_indices() {
    local indices=""
    IFS=',' read -ra uuid_array <<< "$CUDA_VISIBLE_DEVICES"
    
    for i in "${!uuid_array[@]}"; do
        uuid="${uuid_array[$i]}"
        
        if [[ "$uuid" == MIG-* ]]; then
            # MIG devices are typically assigned sequential indices starting from 0
            indices="${indices:+$indices,}$i"
        elif [[ "$uuid" == GPU-* ]]; then
            # Handle regular GPU devices
            clean_uuid="${uuid#GPU-}"
            local index=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | \
                         grep "$clean_uuid" | cut -d',' -f1 | tr -d ' ')
            if [[ -n "$index" ]]; then
                indices="${indices:+$indices,}$index"
            fi
        else
            # Fallback for devices without prefix
            indices="${indices:+$indices,}$i"
        fi
    done
    
    echo "$indices"
}
