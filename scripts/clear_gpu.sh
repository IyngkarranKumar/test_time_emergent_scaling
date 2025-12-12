#!/bin/bash

echo "Clearing GPU memory..."

# Find and kill specific GPU processes (excluding current shell)
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | while read pid; do
    if [ "$pid" != "$$" ] && [ -n "$pid" ]; then
        echo "Killing GPU process PID: $pid"
        kill -9 "$pid" 2>/dev/null || true
    fi
done

# Reset NVIDIA GPU state
nvidia-smi --gpu-reset 2>/dev/null || true

# Clear any lingering CUDA contexts
echo "Resetting GPU state..."
sleep 2

# Show final GPU status
echo "GPU status after cleanup:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

echo "GPU memory cleared!"