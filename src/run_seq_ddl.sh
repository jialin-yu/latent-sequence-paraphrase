#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu
#SBATCH --mem=24g
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH --job-name=test_env
#SBATCH --time=2-0
#SBATCH --nodelist=gpu11

source ../env/bin/activate
module load cuda/11.3

python run_seq.py \
    --data='mscoco' \
    --seed=8000 \
    --un_train_size=115000 \
    --train_size=115000 \
    --seq2seq_lr=0.001 \
    --n_lays=2 \
    --seq2seq_max_epoch=20 \
    --duo=True \
    --dropout=0.3 \


