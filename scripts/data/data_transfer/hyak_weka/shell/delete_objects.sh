#!/bin/bash
#SBATCH --job-name=delete_obj
#SBATCH --output="slurm_job_output/delete_obj_%A_%a.out"
#SBATCH --chdir=/mmfs1/gscratch/efml/hvn2002/open_whisper
#SBATCH --partition=ckpt
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=50
#SBATCH --time=04:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu
#SBATCH --requeue

source ~/.bashrc
conda activate open_whisper

cat $0
echo "--------------------"

python delete_objects.py