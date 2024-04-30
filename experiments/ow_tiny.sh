#!/bin/bash
#SBATCH --job-name=ow_tiny
#SBATCH --partition=gpu-a40
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --gres=gpu:5
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu

# I use source to initialize conda into the right environment.
source ~/.bashrc
conda activate open_whisper
cd /mmfs1/gscratch/efml/hvn2002/open_whisper

cat $0
echo "--------------------"

torchrun --nnodes 1 --nproc_per_node 5 scripts/training/train.py \
    --model_variant=tiny \
    --exp_name=ow_tiny \
    --job_type=train \
    --filter=baseline \
    --run_id=None \
    --rank=None \
    --world_size=None \
    --lr=1.5e-3 \
    --betas="(0.9, 0.98)" \
    --eps=1e-6 \
    --weight_decay=0.1 \
    --max_grad_norm=1.0 \
    --subset=5000 \
    --epochs=8 \
    --eff_size=256 \
    --train_batch_size=32 \
    --val_batch_size=32 \
    --eval_batch_size=32 \
    --train_val_split=0.99 \
    --num_workers=40 \
    --pin_memory=True \
    --shuffle=True \
    --persistent_workers=True \
    --run_eval=True