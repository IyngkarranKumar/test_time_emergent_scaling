#! /bin/bash
#profile device disk space to prevent os memory errors

#!/bin/bash

# Simple disk space profiler
# Usage: ./disk_profiler.sh [directory]

DISK=$(df . | grep -v "Filesystem" | awk '{print $1}')
DIR="${1:-.}"

echo "=== Disk Usage Summary ==="
df -h . | grep -v "Filesystem" #usage of disk that CWD is in (claude written)

echo "=== Largest directories in ./hf_cache ==="
du -h -d 1 ./hf_cache/* 2>/dev/null | sort -hr | head -n 10


echo "Clearing pip and conda caches..."
pip cache purge
conda clean --all -y

echo "=== Disk Usage Summary ==="
df -h . | grep -v "Filesystem" #usage of disk that CWD is in (claude written)
