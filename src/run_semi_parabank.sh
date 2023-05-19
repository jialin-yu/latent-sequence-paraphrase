#!/bin/bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu
#SBATCH --mem=20g
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH --job-name=semi-quora
#SBATCH --time=7-0
#SBATCH --nodelist=gpu2

source ../env/bin/activate
module load cuda/11.3


python run_semi.py \
    --data='quora' \
    --seed=1000 \
    --un_train_size=100000 \
    --train_size=50000 \
    --semi_lr=0.0002 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=True \
    --top_k=10 \
    --beta=1 \

python run_semi.py \
    --data='quora' \
    --seed=2000 \
    --un_train_size=100000 \
    --train_size=50000 \
    --semi_lr=0.0002 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=True \
    --top_k=10 \
    --beta=1 \

python run_semi.py \
    --data='quora' \
    --seed=3000 \
    --un_train_size=100000 \
    --train_size=50000 \
    --semi_lr=0.0002 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=True \
    --top_k=10 \
    --beta=1 \

python run_semi.py \
    --data='quora' \
    --seed=1000 \
    --un_train_size=50000 \
    --train_size=50000 \
    --semi_lr=0.0002 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=True \
    --top_k=10 \
    --beta=1 \

python run_semi.py \
    --data='quora' \
    --seed=2000 \
    --un_train_size=50000 \
    --train_size=50000 \
    --semi_lr=0.0002 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=True \
    --top_k=10 \
    --beta=1 \

python run_semi.py \
    --data='quora' \
    --seed=3000 \
    --un_train_size=50000 \
    --train_size=50000 \
    --semi_lr=0.0002 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=True \
    --top_k=10 \
    --beta=1 \

python run_semi.py \
    --data='quora' \
    --seed=1000 \
    --un_train_size=100000 \
    --train_size=20000 \
    --semi_lr=0.0002 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=True \
    --top_k=10 \
    --beta=1 \

python run_semi.py \
    --data='quora' \
    --seed=2000 \
    --un_train_size=100000 \
    --train_size=20000 \
    --semi_lr=0.0002 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=True \
    --top_k=10 \
    --beta=1 \

python run_semi.py \
    --data='quora' \
    --seed=3000 \
    --un_train_size=100000 \
    --train_size=20000 \
    --semi_lr=0.0002 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=True \
    --top_k=10 \
    --beta=1 \

python run_semi.py \
    --data='quora' \
    --seed=1000 \
    --un_train_size=20000 \
    --train_size=20000 \
    --semi_lr=0.0002 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=True \
    --top_k=10 \
    --beta=1 \

python run_semi.py \
    --data='quora' \
    --seed=2000 \
    --un_train_size=20000 \
    --train_size=20000 \
    --semi_lr=0.0002 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=True \
    --top_k=10 \
    --beta=1 \

python run_semi.py \
    --data='quora' \
    --seed=3000 \
    --un_train_size=20000 \
    --train_size=20000 \
    --semi_lr=0.0002 \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    --use_lm=True \
    --top_k=10 \
    --beta=1 \









python run_seq.py \
    --data='quora' \
    --seed=1000 \
    --un_train_size=50000 \
    --train_size=50000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=False \

python run_seq.py \
    --data='quora' \
    --seed=2000 \
    --un_train_size=50000 \
    --train_size=50000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=False \

python run_seq.py \
    --data='quora' \
    --seed=3000 \
    --un_train_size=50000 \
    --train_size=50000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=False \

python run_seq.py \
    --data='quora' \
    --seed=1000 \
    --un_train_size=50000 \
    --train_size=20000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=False \

python run_seq.py \
    --data='quora' \
    --seed=2000 \
    --un_train_size=50000 \
    --train_size=20000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=False \

python run_seq.py \
    --data='quora' \
    --seed=3000 \
    --un_train_size=50000 \
    --train_size=20000 \
    --seq2seq_lr=0.0002 \
    --seq2seq_max_epoch=30 \
    --seq2seq=False \