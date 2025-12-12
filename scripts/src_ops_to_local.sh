#!bin/bash

# Python-style list of job IDs
JOB_LIST="['50954043', '50936733', '50984902', '51091497', '50961590', '51071401', '50953277', '50982189', '51079847']"

REMOTE_HOST="eddie_ecdf"
REMOTE_DIR="budget_forcing_emergence/gridengine_logs/"
LOCAL_DIR="./src_ops_logs/"

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Create temporary filter file
FILTER_FILE=$(mktemp)
trap "rm -f $FILTER_FILE" EXIT

# Parse Python list - remove brackets, quotes, split on comma
JOB_IDS=$(echo "$JOB_LIST" | tr -d "[]'" | tr ',' '\n' | tr -d ' ')

# Build filter rules
for job_id in $JOB_IDS; do
    echo "+ $job_id/***" >> "$FILTER_FILE"
done
echo "- *" >> "$FILTER_FILE"

# Run rsync
echo "Syncing job directories from $REMOTE_HOST..."
rsync -avz --filter=". $FILTER_FILE" "$REMOTE_HOST:$REMOTE_DIR" "$LOCAL_DIR"

echo "Sync complete!"