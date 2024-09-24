#!/bin/bash
#SBATCH --job-name=remove_faulty
#SBATCH --output="slurm_job_output/remove_faulty_%A_%a.out"
#SBATCH --chdir=/mmfs1/gscratch/efml/hvn2002/open_whisper
#SBATCH --partition=gpu-a40
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=20
#SBATCH --time=04:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu

source ~/.bashrc
conda activate open_whisper
cd /mmfs1/gscratch/efml/hvn2002/open_whisper

cat $0
echo "--------------------"

python remove_faulty_text.py --tar_paths=/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/014533.tar