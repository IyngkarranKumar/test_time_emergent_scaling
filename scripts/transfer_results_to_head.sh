#scratch_disk_dir=/exports/eddie/scratch/s2517451
SOURCE_PATH=${scratch_disk_dir}/data
TARGET_PATH=${PWD}/results_data

#spec_path=Qwen2.5-3B-Instruct_gsm8k_08-05_09-53-21
#SOURCE_PATH=/disk/scratch_fast/s2517451/data/${spec_path}
#TARGET_PATH=${PWD}/results_data/${spec_path}

find "${SOURCE_PATH}" -type d -name "*DeepSeek*" -exec rsync -avz --ignore-existing {} "${TARGET_PATH}/" \;
find "${SOURCE_PATH}" -type d -name "*QwQ*" -exec rsync -avz --ignore-existing {} "${TARGET_PATH}/" \;
find "${SOURCE_PATH}" -type d -name "*Phi*" -exec rsync -avz --ignore-existing {} "${TARGET_PATH}/" \;


