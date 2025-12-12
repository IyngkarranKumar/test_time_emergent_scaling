#!/bin/bash

#nodes with cache
damnii08, damnii02 
crannog04, crannog03 
scotia01

#gpu info
sinfo -o "%P %N %G %f" | grep -v "GRES=N/A" #mlp
qstat -F gpu,gpu-mig,gputype,h_vmem -q gpu #eddie

#A100 quick session 
qlogin -q gpu -l gpu=1 -l h_rt=01:00:00
qlogin -q gpu -l gpu-mig=2 -l h_rt=01:00:00 #MIG GPUs
source /exports/applications/support/set_qlogin_environment.sh  #must run when on intractive sessions

#onto particular eddie node 
qlogin -q gpu -l h_rt=00:15:00 -pe sharedmem 1 -l h_vmem=2G -l hostname=node3b03.ecdf.ed.ac.uk

#l40 quick session 
srun --partition=PGR-Standard --nodelist=scotia04 --time=00:30:00 --cpus-per-task=1 --mem=1G --pty bash

#onto damniis
srun --nodelist=damnii02 --partition=PGR-Standard --gres=gpu:4 --time=2:00:00 --pty bash
srun --nodelist=damnii08 --partition=PGR-Standard --gres=gpu:4 --time=2:00:00 --pty bash
srun --nodelist=damnii07 --partition=PGR-Standard --gres=gpu:2 --time=2:00:00 --pty bash

#onto gpu nodes with A6000s
: '
srun --partition=Teach-Standard \
     --nodelist=landonia[04-07,25] \
     --time=2:00:00 \
     --gres=gpu:a6000:2 \
     --pty bash
'

#rtx 2080tis
srun --partition=PGR-Standard --time=2:00:00 --nodelist=damnii02 --gres=gpu:rtx_2080_ti:2 --pty bash

#a40s
srun --partition=PGR-Standard --time=2:00:00  --gres=gpu:a40:4 --pty bash

#a6000s
srun --partition=Teach-Standard --time=2:00:00 --gres=gpu:a6000:2 --pty bash

#l40s
srun --partition=PGR-Standard --time=2:00:00 --gres=gpu:l40s:2 --pty bash

srun --partition=PGR-Standard --time=00:30:00 --nodelist=scotia01 --cpus-per-task=1 --mem=1G --pty bash
#h200
srun --partition=PGR-Standard-Noble --time=00:30:00 --gres=gpu:h200:1 --pty bash


#get gpu info + partition 
sinfo -o "%P %N %G %f" | grep -v "GRES=N/A"



##common runs 
python3 main.py --config_file config/gsm8k.yaml

#clean results data 
# First, see what would be deleted:
ls -d results_data/*/ | grep -v "_08-"

# If it looks right, then delete:
ls -d results_data/*/ | grep -v "_08-" | xargs -I {} rm -rf "{}"

#kill processes 
pkill -9 -f "main.py" #-9 flag is SIGKILL - the nuclear option

#for deletion

# To remove directories starting with "Qwen2.5"
find $scratch_disk_dir/data/ -type d -name "Qwen2.5*" -ls

# If you want to remove them:
find $scratch_disk_dir/data/ -type d -name "Qwen2.5*" -exec rm -rf {} \;




#debug eddie_ecdf

rm -rf ~/.cursor-server/
find ~/.cursor-server/data/logs/ -name "*.log" -exec tail -20 {} \;
echo 'export NODE_OPTIONS="--max-old-space-size=1024 --max-semi-space-size=32"' >> ~/.bashrc

#remove those with less than N files
find results_data -type d -exec bash -c 'shopt -s nullglob dotglob; files=("$1"/*); [ ${#files[@]} -lt 3 ] && echo "$1"' _ {} \;

#remoe those before given date
ls -d *_07-*_* *_09-0[1-9]_*

#copying logs across

#finding large logs
find /home/s2517451/budget_forcing_emergence/logs -name "*.log" -size +500k -exec du -sh {} \; | sort -hr

rsync -av --include='*.log' --include='*/' --exclude='*' --min-size=500k \
mlp_head:/home/s2517451/budget_forcing_emergence/logs/ logs_to_analyse/


rsync -av --include='*.log' --include='*/' --exclude='*' --min-size=500k \
eddie_ecdf:/home/s2517451/budget_forcing_emergence/logs/ logs_to_analyse/


rsync -avz --progress /data/results_data/ /home/eidf029/eidf029/s2517451-infk8s/budget_forcing_emergence/results_data/

kubectl rsync -avz s2517451-infk8s-outputs-rsync-backend-sjtr6:/data/results_data/ /home/eidf029/eidf029/s2517451-infk8s/budget_forcing_emergence/results_data/
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 && python3 results_debug.py

kubectl attach -it <pod-name> -c test


find ~/budget_forcing_emergence/main_results_data/Deepseek-R1-Distill-Qwen-1.5B_gpqa -name "*.pkl" -exec ls -lh {} \

#login node
ls -lh ~/budget_forcing_emergence/main_results_data/DeepSeek-R1-Distill-Qwen-1.5B_gpqa/*.pkl | awk '{n=split($NF,a,"/"); print $5, a[n-1]"/"a[n]}' | sort -k1,1nr

#workspace pvc 
ls -lh main_results_data/DeepSeek-R1-Distill-Qwen-1.5B_gpqa/*.pkl | awk '{n=split($NF,a,"/"); print $5, a[n-1]"/"a[n]}' | sort -k1,1nr

#outputs pvc
ls -lh /data/results_data/DeepSeek-R1-Distill-Qwen-1.5B_gpqa*/*.pkl | awk '{n=split($NF,a,"/"); print $5, a[n-1]"/"a[n]}' | sort -k1,1nr

#checking checksums 
md5sum main_results_data/Deepseek-R1-Distill-Qwen-1.5B_gpqa/save_file__budget_8192_samples_60-84.pkl
md5sum DeepSeek-R1-Distill-Qwen-1.5B_gpqa_10-25_14-34-15/save_file__budget_8192_samples_60-84.pkl
md5sum main_results_data/Deepseek-R1-Distill-Qwen-1.5B_gpqa/save_file__budget_8192_samples_60-84.pkl


# rsync port-forwarding 
apt-get update && apt-get install -y openssh-server rsync
mkdir -p /run/sshd
echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
passwd  # set password
/usr/sbin/sshd -D

kubectl port-forward s2517451-infk8s-outputs-rsync-backend-4xrm6 3333:22 #start tunnel 

pattern="DeepSeek-R1-Distill-Qwen-1.5B_gpqa"

#go from otputs pvc to login node 
pattern="DeepSeek-R1-Distill-Qwen-32B_gpqa"
rsync -avzc --partial --progress \
  --include="${pattern}*/" --include="${pattern}*/**" --exclude="*" \
  -e "ssh -p 3333 -o StrictHostKeyChecking=no" \
  root@localhost:/data/results_data/ \
  ~/budget_forcing_emergence/results_data/

# go from login node to workspace pvc 
directory="DeepSeek-R1-Distill-Qwen-32B_gpqa"
rsync -avzc --partial --progress \
  -e "ssh -p 4444 -o StrictHostKeyChecking=no" \
  ~/budget_forcing_emergence/main_results_data/${directory}/ \
  root@localhost:/data/budget_forcing_emergence/main_results_data/${directory}/

pattern="DeepSeek-R1-Distill-Qwen-1.5B_gpqa"
hashes=$(checksum_pattern $pattern)
echo "$hashes"