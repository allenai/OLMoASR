
#!/bin/bash

# Enable verbose mode and redirect all output to a log file
exec > >(tee -a startup.log) 2>&1
set -x

# Commands
echo "This will be logged to the file"

# gsutil cp gs://ow-seg/preprocess_gcs.py /home/huongn
gsutil cp gs://ow-seg/requirements_seg.txt /home/huongn
# gsutil cp gs://ow-seg/utils.py /home/huongn
# gsutil cp gs://ow-seg/merge_man_mach.py /home/huongn
gsutil cp gs://ow-seg/merge_man_mach_neat.py /home/huongn

apt-get update && apt-get install -y     curl     git     ffmpeg     python3     python3-venv     python3-pip     python3-setuptools     unzip     curl

python3 -m venv /home/huongn/venv
source /home/huongn/venv/bin/activate

# Add the virtual environment activation to .bashrc to ensure it's activated on login
echo "source /home/huongn/venv/bin/activate" >> /home/huongn/.bashrc
# Add the virtual environment activation to .profile
echo "source /home/huongn/venv/bin/activate" >> /home/huongn/.profile

source /home/huongn/.bashrc
source /home/huongn/.profile

# Install Python requirements
pip install -r /home/huongn/requirements_seg.txt

# Download the AWS CLI installation package to /home/huongn
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/home/huongn/awscliv2.zip"

# Unzip the package in /home/huongn
unzip /home/huongn/awscliv2.zip -d /home/huongn

# Run the install script from /home/huongn
/home/huongn/aws/install

# Define the username and the limits you want to set
USERNAME="huongn"
HARD_LIMIT=1000000

# Backup the original /etc/security/limits.conf file
sudo cp /etc/security/limits.conf /etc/security/limits.conf.bak

# Update limits.conf if the limits for the user haven't been set
if ! grep -q "$USERNAME hard nofile" /etc/security/limits.conf; then
  echo "$USERNAME hard nofile $HARD_LIMIT" | sudo tee -a /etc/security/limits.conf
fi

echo "Limits updated successfully."

# Verify the installation of AWS CLI
aws --version

# AWS configuration
AWS_ACCESS_KEY_ID="AKIAZW3TMCLLUA6MAQLI"
AWS_SECRET_ACCESS_KEY="9WmLHXghdPB8AVDQ3GEhfSmn85eurvsr5yNLIg//"
AWS_DEFAULT_REGION="us-west-1"
AWS_OUTPUT_FORMAT="json"  # Options: json, text, table
AWS_PROFILE="default"  # Replace with your profile name if needed

# Ensure the .aws directory exists in /home/huongn
mkdir -p /home/huongn/.aws

# Write to AWS credentials file in /home/huongn
cat <<EOL > /home/huongn/.aws/credentials
[$AWS_PROFILE]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
EOL

# Write to AWS config file in /home/huongn
cat <<EOL > /home/huongn/.aws/config
[$AWS_PROFILE]
region = $AWS_DEFAULT_REGION
output = $AWS_OUTPUT_FORMAT
EOL

cd /home/huongn
# python3 merge_man_mach.py >> main.log 2>&1
python3 merge_man_mach_neat.py >> main.log 2>&1
# python3 preprocess_gcs.py --bucket=ow-seg --queue_id=ow-seg --log_dir=seg_logs --seg_dir=segments --transcript_only=False --audio_dir=audio --tar_prefix=ow_4M_full --jsonl_prefix=None --manifest_prefix=None  >> main.log 2>&1
