GDRIVE_FOLDER_ID=1oPSqZ45J-FlP6ZxQBXfz0rYsSbCW7sLy
SRC_PATH=./results_data
DST_PATH=gdrive:/{$GDRIVE_FOLDER_ID}

# Copy contents of source directory to Google Drive destination
rclone copy "$SRC_PATH" "$DST_PATH" \
  --progress \
  --transfers 4 \
  --checkers 8 \
  --tpslimit 10 \
  --stats 1s \
