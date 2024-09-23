#!/bin/bash
#SBATCH --job-name=gen_shards_json.sh
#SBATCH --output=slurm_job_output/gen_shards_json_%A_%a.out
#SBATCH --partition=gpu-a40
#SBATCH --account=efml
#SBATCH --nodes=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=30
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu
#SBATCH --array=0-104
#SBATCH --requeue

# I use source to initialize conda into the right environment.
source ~/.bashrc
conda activate open_whisper
cd /mmfs1/gscratch/efml/hvn2002/open_whisper

cat $0
echo "--------------------"



python scripts/data/data_transfer/gen_shards_json.py bulk_encode 
