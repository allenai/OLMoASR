#!/bin/bash
#SBATCH --job-name=ow_tiny
#SBATCH --partition=gpu-a40
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --mem=180G
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-gpu=9
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu

# I use source to initialize conda into the right environment.
source ~/.bashrc
conda activate open_whisper
cd /mmfs1/gscratch/efml/hvn2002/open_whisper

cat $0
echo "--------------------"

torchrun --nnodes 1 --nproc_per_node 1 scripts/training/train_wds.py \
    --model_variant=tiny \
    --exp_name=ow_tiny_wds \
    --job_type=train \
    --train_shards=data/tars/\{000000..000001\}.tar \
    --val_shards=data/tars/\{000002..000003\}.tar \
    --len_train_data=None \
    --len_val_data=None \
    --run_id=None \
    --rank=None \
    --world_size=None \
    --lr=1.5e-3 \
    --betas="(0.9, 0.98)" \
    --eps=1e-6 \
    --weight_decay=0.1 \
    --max_grad_norm=1.0 \
    --epochs=2 \
    --eff_size=256 \
    --train_batch_size=8 \
    --val_batch_size=8 \
    --eval_batch_size=8 \
    --num_workers=45 \
    --pin_memory=True \
    --persistent_workers=True \
    --run_eval=False