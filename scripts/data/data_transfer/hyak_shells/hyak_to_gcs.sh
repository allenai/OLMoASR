#!/bin/bash
#SBATCH --job-name=hyak_to_gcs_1
#SBATCH --output="slurm_job_output/hyak_to_gcs/hyak_to_gcs_1_%A_%a.out"
#SBATCH --partition=ckpt
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=30
#SBATCH --time=04:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu
#SBATCH --array=679-699
#SBATCH --requeue

TARS_PATH="/mmfs1/gscratch/efml/hvn2002/ow_440K_wds"
JOB_INDEX=$SLURM_ARRAY_TASK_ID
BATCH_SIZE=74

source ~/.bashrc
conda activate open_whisper
cd /mmfs1/gscratch/efml/hvn2002/open_whisper

cat $0
echo "--------------------"

python scripts/data/data_transfer/hyak_to_gcs.py bulk_upload --tar_dir=$TARS_PATH \
    --job_index=$JOB_INDEX \
    --batch_size=$BATCH_SIZE