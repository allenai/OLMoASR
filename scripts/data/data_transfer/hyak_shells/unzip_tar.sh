#!/bin/bash
#SBATCH --job-name=unzip_tar
#SBATCH --output=slurm_job_output/slurm-%j.out
#SBATCH --partition=gpu-a40
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu

# I use source to initialize conda into the right environment.
source ~/.bashrc
conda activate open_whisper
cd /mmfs1/gscratch/efml/hvn2002/open_whisper

cat $0
echo "--------------------"

python scripts/data/data_transfer/unzip_tar.py --tar_files=00001651.tar.gz,00002260.tar.gz,00001230.tar.gz,00001829.tar.gz,00002148.tar.gz,00000224.tar.gz,00002014.tar.gz,00001538.tar.gz,00000616.tar.gz,00000093.tar.gz,00000619.tar.gz,00002158.tar.gz,00001898.tar.gz,00001134.tar.gz,00000898.tar.gz,00000678.tar.gz,00000656.tar.gz,00002391.tar.gz,00001931.tar.gz,00001292.tar.gz,00001548.tar.gz,00000113.tar.gz,00000089.tar.gz,00001694.tar.gz,00000866.tar.gz,00001662.tar.gz,00002087.tar.gz,00001243.tar.gz,00000229.tar.gz,00001742.tar.gz,00001597.tar.gz,00000755.tar.gz,00001467.tar.gz,00000338.tar.gz,00000770.tar.gz,00000881.tar.gz,00000295.tar.gz,00001677.tar.gz,00001094.tar.gz,00000161.tar.gz,00000394.tar.gz,00002034.tar.gz,00000736.tar.gz,00002304.tar.gz,00000875.tar.gz,00000269.tar.gz,00000219.tar.gz,00001458.tar.gz,00000698.tar.gz,00000488.tar.gz,00002343.tar.gz,00002428.tar.gz,00002225.tar.gz,00000080.tar.gz,00000752.tar.gz,00000364.tar.gz,00001833.tar.gz,00000310.tar.gz,00000425.tar.gz,00000352.tar.gz,00000838.tar.gz,00001500.tar.gz,00001904.tar.gz,00001530.tar.gz,00002080.tar.gz,00001937.tar.gz,00002089.tar.gz,00000082.tar.gz,00002378.tar.gz,00001109.tar.gz,00000575.tar.gz,00000769.tar.gz,00000939.tar.gz,00001251.tar.gz,00001040.tar.gz,00000162.tar.gz,00000653.tar.gz,00002345.tar.gz,00002448.tar.gz,00001901.tar.gz