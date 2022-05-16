#!/bin/bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu
#SBATCH --mem=20g
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH --job-name=semi-no-lm
#SBATCH --time=7-0
#SBATCH --nodelist=gpu10

source ../env/bin/activate
module load cuda/11.3

python run_semi.py \
    --data='quora' \
    --seed=1000 \
    --un_train_size=100000 \
    --train_size=100000 \
    --semi_lr=0.0002 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=False \
    --top_k=10 \
    --beta=1 \

python run_semi.py \
    --data='quora' \
    --seed=2000 \
    --un_train_size=100000 \
    --train_size=100000 \
    --semi_lr=0.0002 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=False \
    --top_k=10 \
    --beta=1 \

python run_semi.py \
    --data='quora' \
    --seed=3000 \
    --un_train_size=100000 \
    --train_size=100000 \
    --semi_lr=0.0002 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=False \
    --top_k=10 \
    --beta=1 \

python run_semi.py \
    --data='mscoco' \
    --seed=1000 \
    --un_train_size=93000 \
    --train_size=93000 \
    --semi_lr=0.0001 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=False \
    --top_k=10 \
    --beta=1 \

python run_semi.py \
    --data='mscoco' \
    --seed=2000 \
    --un_train_size=93000 \
    --train_size=93000 \
    --semi_lr=0.0001 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=False \
    --top_k=10 \
    --beta=1 \

python run_semi.py \
    --data='mscoco' \
    --seed=3000 \
    --un_train_size=93000 \
    --train_size=93000 \
    --semi_lr=0.0001 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=False \
    --top_k=10 \
    --beta=1 \
