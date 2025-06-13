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
sudo yum install python3-devel -y
aws configure set aws_access_key_id AKIASHLPW4FE63DTIAPD #<---- fill in here 
aws configure set aws_secret_access_key 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 #<--- fill in here
aws configure set default.region us-east-1
wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz 
tar -xvzf s5cmd_2.2.2_Linux-64bit.tar.gz 
sudo yum install git -y 
sudo yum install pip -y

# get the data
mkdir unfiltered_jsonls
mkdir filtered_jsonls
/mnt/raid0/s5cmd cp "s3://allennlp-mattj/openwhisper/pretraining_data/UNFILTERED_JSONLS/*" "unfiltered_jsonls/" # <--- fill in UNFILTERED_JSONLS
/mnt/raid0/s5cmd cp "s3://allennlp-mattj/openwhisper/filter_configs/*" "./" # <--- fill in UNFILTERED_JSONLS

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

source /home/ec2-user/.bashrc
source /home/ec2-user/.profile

echo 'export PYTHONPATH=/mnt/raid0/open_whisper:$PYTHONPATH' >> /home/ec2-user/.bashrc
source /home/ec2-user/.bashrc

# install the requirements
pip install wheel
pip install -r open_whisper/requirements/requirements-data.txt
python -m spacy download en_core_web_sm

# run the script
python3 open_whisper/scripts/data/filtering/data_filter.py --config=CONFIG --input-dir=INPUT_DIR --output-dir=OUTPUT_DIR
# python3 open_whisper/scripts/data/filtering/seg_data_filter.py --config=CONFIG --input-dir=INPUT_DIR --output-dir=OUTPUT_DIR

/mnt/raid0/s5cmd cp "filtered_jsonls/*" "s3://allennlp-mattj/openwhisper/pretraining_data/text_heurs_1_manmach_0.8_jan_25/"