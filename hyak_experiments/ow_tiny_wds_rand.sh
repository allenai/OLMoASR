#!/bin/bash
#SBATCH --job-name=ow_tiny_wds_rand
#SBATCH --output=slurm_job_output/training/ow_tiny_wds_rand_%A_%a.out
#SBATCH --partition=gpu-a40
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=4
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu

# I use source to initialize conda into the right environment.
source ~/.bashrc
conda activate open_whisper
cd /mmfs1/gscratch/efml/hvn2002/open_whisper

cat $0
echo "--------------------"

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

torchrun --nnodes 1 --nproc_per_node 1 --master-port=29504 scripts/training/train_wds_no_reslog_random.py \
    --model_variant=tiny \
    --exp_name=ow_tiny_random \
    --job_type=train \
    --train_shards=/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/\{000000..000007\}.tar \
    --train_steps=1048576 \
    --val_shards=/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/073468.tar \
    --run_id=None \
    --ckpt_file_name=None \
    --log_dir=logs \
    --eval_dir=data/eval \
    --rank=None \
    --world_size=None \
    --lr=1.5e-3 \
    --betas="(0.9, 0.98)" \
    --eps=1e-6 \
    --weight_decay=0.1 \
    --max_grad_norm=1.0 \
    --eff_batch_size=256 \
    --train_batch_size=8 \
    --val_batch_size=8 \
    --eval_batch_size=8 \
    --num_workers=4 \
    --pin_memory=True \
    --persistent_workers=True \
    --run_val=False \
    --run_eval=False