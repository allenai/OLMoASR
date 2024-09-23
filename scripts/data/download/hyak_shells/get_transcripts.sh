#!/bin/bash
#SBATCH --job-name=get_transcripts
#SBATCH --partition=gpu-a40
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=30
#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu
# I use source to initialize conda into the right environment.
source ~/.bashrc
conda activate open_whisper
cd /mmfs1/gscratch/efml/hvn2002/open_whisper

cat $0
echo "--------------------"

python scripts/data/preprocess/download_transcripts.py --samples_file=logs/data/download/sampled_en_05-02.txt