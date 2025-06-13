#!/bin/bash

sudo yum install mdadm -y
sudo mdadm --create /dev/md0 --level=0 --raid-devices=8 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 /dev/nvme5n1 /dev/nvme6n1 /dev/nvme7n1 /dev/nvme8n1 
sudo mkfs.xfs /dev/md0
sudo mkdir /mnt/raid0
sudo mount /dev/md0 /mnt/raid0
sudo chown -R $USER /mnt/raid0
cd /mnt/raid0

sudo yum install gcc -y
sudo yum install cmake -y
sudo yum install openssl-devel -y
sudo yum install g++ -y
sudo yum install htop -y
aws configure set aws_access_key_id AKIASHLPW4FE63DTIAPD #<---- fill in here 
aws configure set aws_secret_access_key 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 #<--- fill in here
aws configure set default.region us-east-1
wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz 
tar -xvzf s5cmd_2.2.2_Linux-64bit.tar.gz 
sudo yum install git -y 
sudo yum install pip -y

mkdir seg_jsonls
/mnt/raid0/s5cmd cp "s3://allennlp-mattj/openwhisper/pretraining_data/SEG_DOLMA_DIR/*" "seg_jsonls/documents/" # <--- fill in FULL_JSONL_DIR
/mnt/raid0/s5cmd cp "s3://allennlp-mattj/openwhisper/fasttext_models/*" "./"

# get the code
GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxx" # Replace with your token
GITHUB_USER="huongngo-8"
REPO="open_whisper"
BRANCH="your-branch-name"  # Replace with the branch you want to clone

# Clone the specific branch
git clone --branch $BRANCH --single-branch https://$GITHUB_USER:$GITHUB_TOKEN@github.com/$GITHUB_USER/$REPO.git

# create virtual environment
python3 -m venv /mnt/raid0/env
source /mnt/raid0/env/bin/activate

# Add the virtual environment activation to .bashrc to ensure it's activated on login
echo "source /mnt/raid0/env/bin/activate" >> /home/ec2-user/.bashrc
# Add the virtual environment activation to .profile
echo "source /mnt/raid0/env/bin/activate" >> /home/ec2-user/.profile
export PYTHONPATH="/mnt/raid0/open_whisper"

source /home/ec2-user/.bashrc
source /home/ec2-user/.profile

pip install uv
uv pip install dolma
pip install -r open_whisper/requirements/requirements-data.txt
pip install transformers==4.38.0

# run the dolma tagging command
dolma tag --documents /mnt/raid0/seg_jsonls/documents/* --taggers ow-tedlium-quality ow-commonvoice-quality --tagger_modules  /mnt/raid0/open_whisper/scripts/data/filtering/fasttext/inference.py --processes 48
# postprocess dolma documents and attributes
# optional - joining multiple attributes into 1 attribute
python3 open_whisper/scripts/data/filtering/fasttext/postprocess.py --docs_dir=None --attributes_dirs=ATTRIBUTES_DIRS --output_dir=OUTPUT_DIR --mode=join_attributes
# converting dolma documents to original format
python3 open_whisper/scripts/data/filtering/fasttext/postprocess.py --docs_dir=DOCS_DIR --attributes_dirs=ATTRIBUTES_DIRS --output_dir=OUTPUT_DIR --mode=join_docs_and_attributes
# running reservoir sampling
python3 open_whisper/scripts/data/filtering/reservoir_sample.py --input-dir=INPUT_DIR --output-loc=OUTPUT_DIR --keys=fasttext_scores,librispeech_clean --reservoir-size=1000000