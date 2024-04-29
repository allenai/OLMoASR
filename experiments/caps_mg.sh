#!/bin/bash
#SBATCH --job-name=caps_mg
#SBATCH --partition=gpu-2080ti
#SBATCH --account=raivn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=180G
#SBATCH --gres=gpu:6
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu

# I use source to initialize conda into the right environment.
source ~/.bashrc
conda activate open_whisper
cd /mmfs1/gscratch/efml/hvn2002/open_whisper

cat $0
echo "--------------------"

torchrun --nnodes 1 --nproc_per_node 6 scripts/training/train_caps_mg.py \
    --model_variant=tiny \
    --exp_name=caps-mg \
    --job_type=train \
    --rank=None \
    --world_size=None \
    --lr=1.5e-3 \
    --betas="(0.9, 0.98)" \
    --eps=1e-6 \
    --weight_decay=0.1 \
    --max_grad_norm=1.0 \
    --subset=None \
    --epochs=14 \
    --eff_size=256 \
    --train_batch_size=64 \
    --val_batch_size=32 \
    --eval_batch_size=64 \
    --train_val_split=0.99 \
    --num_workers=42 \
    --pin_memory=True \
    --shuffle=True \
    --persistent_workers=True \
    --run_eval=True