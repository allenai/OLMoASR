#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=ckpt
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=4:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu

# I use source to initialize conda into the right environment.
source ~/.bashrc
conda activate open_whisper
cd /mmfs1/gscratch/efml/hvn2002/open_whisper

cat $0
echo "--------------------"
python scripts/data/preprocess/preprocess.py --data_shard_path=data/00000000 --num_output_shards=6