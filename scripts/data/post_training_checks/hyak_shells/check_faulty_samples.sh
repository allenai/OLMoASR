#!/bin/bash
#SBATCH --job-name=check_faulty_samples
#SBATCH --output=slurm_job_output/check_faulty_samples_%A_%a.out
#SBATCH --partition=gpu-a40
#SBATCH --account=raivn
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu

source ~/.bashrc
conda activate open_whisper
cd /mmfs1/gscratch/efml/hvn2002/open_whisper

cat $0
echo "--------------------"

python check_faulty_samples.py --shards=/mmfs1/gscratch/efml/hvn2002/014533.tar --batch_size=256