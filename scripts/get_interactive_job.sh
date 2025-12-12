
echo "Cluster node info"
sinfo -o "%N %c %m %f %G %P %T %D"



echo "Getting interactive job"
srun -p PGR-Standard --gres=gpu:a40:2 --nodelist=crannog07 --time=04:00:00 --pty bash

echo "Getting interactive job"
srun -p PGR-Standard --nodelist=crannog03 --time=01:00:00 --pty bash

#a6000s
srun -p Teach-Standard --gres=gpu:a6000:2 --time=04:00:00 --pty bash


# damnii02 has hf cache setup.
echo "Getting interactive job with 2 RTX 2080s"
srun -p PGR-Standard --gres=gpu:rtx_2080_ti:2 --time=04:00:00 --pty bash

srun -p PGR-Standard -w damnii02 --gres=gpu:rtx_2080_ti:2 --time=04:00:00 --pty bash

