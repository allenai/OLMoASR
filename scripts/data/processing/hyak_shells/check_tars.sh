#!/bin/bash
#SBATCH --job-name=check_tars
#SBATCH --output="slurm_job_output/check_tars/check_tars_%A_%a.out"
#SBATCH --chdir=/mmfs1/gscratch/efml/hvn2002/open_whisper
#SBATCH --partition=ckpt
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=30
#SBATCH --time=04:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu
#SBATCH --array=0-91
#SBATCH --requeue

JOB_BATCH_IDX=$SLURM_ARRAY_TASK_ID

source ~/.bashrc
conda activate open_whisper

cat $0
echo "--------------------"

TAR_PATHS=$(python -c "
import json
import sys
job_batch_idx = int(sys.argv[1])
with open('logs/data/preprocess/check_tars.json', 'r') as f:
    tar_paths_all = json.load(f)
tar_paths = tar_paths_all[f'batch_{job_batch_idx}']
print(tar_paths)
" $JOB_BATCH_IDX)

echo "Job Batch Index: $JOB_BATCH_IDX"

python scripts/data/preprocess/check_tars.py --tar_paths_str=$TAR_PATHS  \
                                            --output_file=logs/data/preprocess/sample_count.txt \
                                            --check=count