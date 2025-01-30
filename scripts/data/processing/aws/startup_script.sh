#!/bin/bash

# Enable verbose mode and redirect all output to a log file
exec > >(tee -a /home/ec2-user/startup.log) 2>&1
set -x

# Commands
echo "This will be logged to the file"

# install some things
sudo yum install gcc -y
sudo yum install cmake -y
sudo yum install openssl-devel -y
sudo yum install g++ -y
sudo yum install htop -y

wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz 
tar -xvzf s5cmd_2.2.2_Linux-64bit.tar.gz 

sudo yum install git -y 
sudo yum install pip -y

apt-get update && apt-get install -y     curl     git     ffmpeg     python3     python3-venv     python3-pip     python3-setuptools     unzip     curl


# Verify the installation of AWS CLI
aws --version

aws configure set aws_access_key_id AKIASHLPW4FE63DTIAPD
aws configure set aws_secret_access_key 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
aws configure set default.region us-east-1

cat ~/.aws/credentials
cat ~/.aws/config

/s5cmd --credentials-file ~/.aws/credentials --profile default cp s3://allennlp-mattj/openwhisper/requirements_gen_jsonl.txt /home/ec2-user
/s5cmd --credentials-file ~/.aws/credentials --profile default cp s3://allennlp-mattj/openwhisper/merge_man_mach.py /home/ec2-user
/s5cmd --credentials-file ~/.aws/credentials --profile default ls s3://allennlp-mattj/openwhisper/

mkdir /home/ec2-user/.aws
cp ~/.aws/credentials /home/ec2-user/.aws/credentials
cp ~/.aws/config /home/ec2-user/.aws/config

python3 -m venv /home/ec2-user/venv
source /home/ec2-user/venv/bin/activate

# Add the virtual environment activation to .bashrc to ensure it's activated on login
echo "source /home/ec2-user/venv/bin/activate" >> /home/ec2-user/.bashrc
# Add the virtual environment activation to .profile
echo "source /home/ec2-user/venv/bin/activate" >> /home/ec2-user/.profile
echo "export PATH=/usr/local/bin:/$PATH" >> /home/ec2-user/.bashrc

source /home/ec2-user/.bashrc
source /home/ec2-user/.profile

pip install -r /home/ec2-user/requirements_gen_jsonl.txt

python3 /home/ec2-user/merge_man_mach.py >> /home/ec2-user/main.log 2>&1
