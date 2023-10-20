#!/bin/bash

#SBATCH -p GPU-shared
#SBATCH -t 03:00:00
#SBATCH --gres=gpu:1

set -x
. /jet/home/jaypat/.bashrc

conda activate odp

cd /jet/home/jaypat/deepreach

srun python experiment_scripts/train_hji_air3D.py --experiment_name experiment_1 --minWith zero  --num_src_samples 10000 --pretrain --pretrain_iters 10000 --num_epochs 120000 --counter_end 110000
