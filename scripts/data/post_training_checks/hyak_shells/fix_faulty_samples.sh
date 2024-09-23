#!/bin/bash
#SBATCH --job-name=reshuffling
#SBATCH --output="slurm_job_output/reshuffling_%A_%a.out"
#SBATCH --chdir=/mmfs1/gscratch/efml/hvn2002/open_whisper
#SBATCH --partition=gpu-a40
#SBATCH --account=efml
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=40
#SBATCH --time=04:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hvn2002@uw.edu

source ~/.bashrc
conda activate open_whisper
cd /mmfs1/gscratch/efml/hvn2002/open_whisper

cat $0
echo "--------------------"

python debug_2.py --tar_paths=/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040560.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040561.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040562.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040563.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040564.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040565.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040566.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040567.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040568.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040569.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040570.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040571.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040572.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040573.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040574.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040575.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040576.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040577.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040578.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040579.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040580.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040581.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040582.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040583.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040584.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040585.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040586.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040587.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040588.tar,/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/040589.tar