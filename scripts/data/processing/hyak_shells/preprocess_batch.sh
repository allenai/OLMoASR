#!/bin/bash
#SBATCH --job-name=preprocess_wds
#SBATCH --output="slurm_job_output/preprocess/preprocess_wds_re_%A_%a.out"
#SBATCH --chdir=/mmfs1/gscratch/efml/hvn2002/open_whisper
#SBATCH --partition=ckpt
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --time=04:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu
#SBATCH --array=0

JOB_BATCH_IDX=1
SHARD_INDEX=$SLURM_ARRAY_TASK_ID

# I use source to initialize conda into the right environment.
source ~/.bashrc
conda activate open_whisper

cat $0
echo "--------------------"

DATA_SHARD_PATH=$(python -c "
import json
import sys
job_batch_idx = int(sys.argv[1])
shard_index = int(sys.argv[2])
with open('logs/data/preprocess/idx_to_batch_re.json', 'r') as f:
    data_paths = json.load(f)
print(data_paths[f'batch_{job_batch_idx}'][shard_index])
" $JOB_BATCH_IDX $SHARD_INDEX)

echo "Job Batch Index: $JOB_BATCH_IDX"
echo "Shard Index: $SHARD_INDEX"
echo "Data Shard Path: $DATA_SHARD_PATH"

python scripts/data/preprocess/preprocess.py --job_batch_idx=$JOB_BATCH_IDX \
                                            --job_idx=$SHARD_INDEX \
                                            --data_shard_path=$DATA_SHARD_PATH \
                                            --num_output_shards=30 \
                                            --in_memory=True
