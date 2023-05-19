#!/bin/bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu
#SBATCH --mem=20g
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH --job-name=seq-trans
#SBATCH --time=7-0
#SBATCH --nodelist=gpu11

source ../env/bin/activate
module load cuda/11.3

python run_seq.py \
    --data='quora' \
    --seed=1000 \
    --un_train_size=50000 \
    --train_size=50000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=True \

python run_seq.py \
    --data='quora' \
    --seed=2000 \
    --un_train_size=50000 \
    --train_size=50000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=True \

python run_seq.py \
    --data='quora' \
    --seed=3000 \
    --un_train_size=50000 \
    --train_size=50000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=True \

python run_seq.py \
    --data='quora' \
    --seed=1000 \
    --un_train_size=20000 \
    --train_size=20000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=True \

python run_seq.py \
    --data='quora' \
    --seed=2000 \
    --un_train_size=20000 \
    --train_size=20000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=True \

python run_seq.py \
    --data='quora' \
    --seed=3000 \
    --un_train_size=20000 \
    --train_size=20000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=True \





python run_seq.py \
    --data='mscoco' \
    --seed=1000 \
    --un_train_size=50000 \
    --train_size=50000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=True \

python run_seq.py \
    --data='mscoco' \
    --seed=2000 \
    --un_train_size=50000 \
    --train_size=50000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=True \

python run_seq.py \
    --data='mscoco' \
    --seed=3000 \
    --un_train_size=50000 \
    --train_size=50000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=True \

python run_seq.py \
    --data='mscoco' \
    --seed=1000 \
    --un_train_size=20000 \
    --train_size=20000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=True \

python run_seq.py \
    --data='mscoco' \
    --seed=2000 \
    --un_train_size=20000 \
    --train_size=20000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=True \

python run_seq.py \
    --data='mscoco' \
    --seed=3000 \
    --un_train_size=20000 \
    --train_size=20000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=True \