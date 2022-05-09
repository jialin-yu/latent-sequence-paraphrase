#!/bin/bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu
#SBATCH --mem=20g
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH --job-name=test_env
#SBATCH --time=7-0
#SBATCH --nodelist=gpu10

source ../env/bin/activate
module load cuda/11.3

python run_semi.py \
    --data='quora' \
    --seed=1111 \
    --un_train_size=50000 \
    --train_size=10000 \
    --semi_lr=0.0002 \
    --semi_max_epoch=30 \
    --fixed_temperature=True \
    --use_lm=False \
    --top_k=5 \
    --beta=0.001 \

python run_semi.py \
    --data='quora' \
    --seed=1111 \
    --un_train_size=50000 \
    --train_size=10000 \
    --semi_lr=0.0002 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=False \
    --top_k=5 \
    --beta=0.001 \