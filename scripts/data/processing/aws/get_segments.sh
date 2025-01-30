#!/bin/bash

# Set up a raid file system on the nitro drives
sudo yum install mdadm -y
sudo mdadm --create /dev/md0 --level=0 --raid-devices=8 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 /dev/nvme5n1 /dev/nvme6n1 /dev/nvme7n1 /dev/nvme8n1 
sudo mkfs.xfs /dev/md0
sudo mkdir /mnt/raid0
sudo mount /dev/md0 /mnt/raid0
sudo chown -R $USER /mnt/raid0
cd /mnt/raid0

# install some things
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

# get the data
mkdir full_jsonls
mkdir seg_manifest
/mnt/raid0/s5cmd cp "s3://allennlp-mattj/openwhisper/pretraining_data/FULL_JSONL_DIR/*" "full_jsonls/" # <--- fill in FULL_JSONL_DIR
/mnt/raid0/s5cmd cp "s3://allennlp-mattj/openwhisper/pretraining_data/SEG_MANIFEST_DIR/*" "seg_manifest/" # <--- fill in SEG_MANIFEST_DIR

# get the code
git clone https://github.com/huongngo-8/open_whisper.git

# create virtual environment
python3 -m venv /mnt/raid0/env
source /mnt/raid0/env/bin/activate

# Add the virtual environment activation to .bashrc to ensure it's activated on login
echo "source /mnt/raid0/env/bin/activate" >> /home/ec2-user/.bashrc
# Add the virtual environment activation to .profile
echo "source /mnt/raid0/env/bin/activate" >> /home/ec2-user/.profile

source /home/ec2-user/.bashrc
source /home/ec2-user/.profile

# install the requirements
pip install -r open_whisper/requirements-data-process.txt

# run the script
python3 open_whisper/scripts/data/processing/segment_jsonl.py --source_dir=full_jsonls --manifest_dir=seg_manifest --log_dir=logs --output_dir=seg_jsonls --subsample=True --subsample_size=2500 --subsample_seed=42

/mnt/raid0/s5cmd cp "logs/*" "s3://allennlp-mattj/openwhisper/pretraining_data/text_heurs_1_jan25_seg/logs/"
/mnt/raid0/s5cmd cp "seg_jsonls/*" "s3://allennlp-mattj/openwhisper/pretraining_data/text_heurs_1_jan25_seg/seg_jsonls/"