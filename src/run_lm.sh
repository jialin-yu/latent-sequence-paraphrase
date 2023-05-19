#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu
#SBATCH --mem=24g
#SBATCH -p res-gpu-small
#SBATCH --qos=long-high-prio
#SBATCH --job-name=test_env
#SBATCH --time=7-0
#SBATCH --nodelist=gpu2

source ../env/bin/activate
module load cuda/11.3

python run_lm.py \
    --data='quora' \
    --semi_max_epoch=30 \
    --fixed_temperature=False \
    
# python run_lm.py \
#     --data='mscoco' \