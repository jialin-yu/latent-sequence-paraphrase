#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu
#SBATCH --mem=24g
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH --job-name=test_env
#SBATCH --time=7-0
#SBATCH --nodelist=gpu11

source ../env/bin/activate
module load cuda/11.3

python run_semi.py \
    --data='quora' \
    --seed=8700 \
    --lm_lr=0.0001 \
    --lm_max_epoch=30 \
    --un_train_size=115000 \
    --train_size=115000 \
    --semi_lr=0.0001 \
    --n_lays=2 \
    --semi_max_epoch=20 \
    --use_pretrain_lm=False\
    --dropout=0.2 \
