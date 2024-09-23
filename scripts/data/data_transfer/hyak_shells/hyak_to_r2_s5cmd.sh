#!/bin/bash
#SBATCH --job-name=hyak_to_r2
#SBATCH --output="slurm_job_output/hyak_to_r2/hyak_to_r2_%A_%a.out"
#SBATCH --chdir=/mmfs1/gscratch/efml/hvn2002/open_whisper
#SBATCH --partition=ckpt
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=20
#SBATCH --time=04:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu
#SBATCH --array=0-1000
#SBATCH --requeue

TARS_PATH="/mmfs1/gscratch/efml/hvn2002/ow_440K_wds"
JOB_INDEX=$SLURM_ARRAY_TASK_ID
BATCH_SIZE=74
LOG_PATH="logs/data/download/hyak_to_r2_s5cmd.jsonl"

source ~/.bashrc
conda activate open_whisper

export S3_ENDPOINT_URL="https://b2bcf985082a37eaf385c532ee37928d.r2.cloudflarestorage.com"

cat $0
echo "--------------------"

python scripts/data/data_transfer/hyak_to_r2_s5cmd.py bulk_upload --tars_path=$TARS_PATH \
    --job_index=$JOB_INDEX \
    --batch_size=$BATCH_SIZE \
    --log_path=$LOG_PATH
