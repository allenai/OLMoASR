#!/bin/bash
#SBATCH --job-name=change_fp
#SBATCH --output="slurm_job_output/change_fp/change_fp_re_%A_%a.out"
#SBATCH --chdir=/mmfs1/gscratch/efml/hvn2002/open_whisper
#SBATCH --partition=ckpt
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --time=04:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu
#SBATCH --array=0-13
#SBATCH --requeue

JOB_BATCH_IDX=1
TAR_PATH_INDEX=$SLURM_ARRAY_TASK_ID

source ~/.bashrc
conda activate open_whisper

cat $0
echo "--------------------"

TAR_PATH=$(python -c "
import json
import sys
job_batch_idx = int(sys.argv[1])
tar_path_index = int(sys.argv[2])
with open('logs/data/preprocess/change_fp_re.json', 'r') as f:
    tar_paths = json.load(f)
print(tar_paths[f'batch_{job_batch_idx}'][tar_path_index])
" $JOB_BATCH_IDX $TAR_PATH_INDEX)

echo "Job Batch Index: $JOB_BATCH_IDX"
echo "Tar Path Index: $TAR_PATH_INDEX"
echo "Tar Path: $TAR_PATH"

python scripts/data/preprocess/change_fp.py --tar_path=$TAR_PATH