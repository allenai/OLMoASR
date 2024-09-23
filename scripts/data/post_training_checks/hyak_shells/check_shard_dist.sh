#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --output=slurm_job_output/shard_dist_%A_%a.out
#SBATCH --partition=gpu-a40
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu

source ~/.bashrc
conda activate open_whisper
cd /mmfs1/gscratch/efml/hvn2002/open_whisper

cat $0
echo "--------------------"

python check_shard_dist.py --shards=/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/\{073468..073469\}.tar