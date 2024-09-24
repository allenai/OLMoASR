#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=ckpt
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --time=4:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu

# I use source to initialize conda into the right environment.
source ~/.bashrc
conda activate open_whisper
cd /mmfs1/gscratch/efml/hvn2002/open_whisper

cat $0
echo "--------------------"
python scripts/data/preprocess/preprocess.py --job_batch_idx=1 \
                                            --job_idx=0 \
                                            --data_shard_path=/mmfs1/gscratch/efml/hvn2002/ow_440K/00000284 \
                                            --num_output_shards=30 \
                                            --in_memory=True