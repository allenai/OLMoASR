#!/bin/bash
#SBATCH --job-name=check_text
#SBATCH --output="slurm_job_output/check_text/check_text_%A_%a.out"
#SBATCH --chdir=/mmfs1/gscratch/efml/hvn2002/open_whisper
#SBATCH --partition=ckpt
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=10
#SBATCH --time=04:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu
#SBATCH --array=9000-9999
#SBATCH --requeue

JOB_BATCH_IDX=$SLURM_ARRAY_TASK_ID

source ~/.bashrc
conda activate open_whisper
cd /mmfs1/gscratch/efml/hvn2002/open_whisper

cat $0
echo "--------------------"

TAR_PATH=$(python -c "
import glob
import sys
all_tar_paths = glob.glob('/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/*')
job_batch_idx = int(sys.argv[1])
tar_path = all_tar_paths[job_batch_idx]
print(tar_path)
" $JOB_BATCH_IDX)

echo "Tar Path: $TAR_PATH"

python debug.py --tar_path=$TAR_PATH --n_text_ctx=448