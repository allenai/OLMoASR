#!/bin/bash
#SBATCH --job-name=check_data_shards
#SBATCH --output="temp/check_data_shards_%A_%a.out"
#SBATCH --chdir=/mmfs1/gscratch/efml/hvn2002/open_whisper
#SBATCH --partition=ckpt
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu
#SBATCH --array=2001-2448

SHARD_INDEX=$SLURM_ARRAY_TASK_ID

source ~/.bashrc
conda activate open_whisper

cat $0
echo "--------------------"
echo "Shard Index: $SHARD_INDEX"

python scripts/data/preprocess/check_data_shards.py --data_shard_idx=$SHARD_INDEX