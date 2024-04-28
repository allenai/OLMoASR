#!/bin/bash
#SBATCH --job-name=caps_mg_exp
#SBATCH --partition=gpu-a40
#SBATCH --account=raivn
#SBATCH --nodes=1
#SBATCH --mem=250G
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-gpu 6
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu

# I use source to initialize conda into the right environment.
source ~/.bashrc
conda activate open_whisper
cd /mmfs1/gscratch/efml/hvn2002/open_whisper

cat $0
echo "--------------------"

python scripts/training/train_w_eval_caps_mg.py
