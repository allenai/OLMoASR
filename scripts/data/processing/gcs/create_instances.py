# from action-atlas
import subprocess
import logging
import sys
sys.path.append("scripts/data/processing/gcs")
from vm_utils import bulk_create_spot_instances
from fire import Fire
from typing import Optional
from google.cloud import storage

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.FileHandler("./create_instances.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

STARTUP_SCRIPT = """
#!/bin/bash

# Enable verbose mode and redirect all output to a log file
exec > >(tee -a startup.log) 2>&1
set -x

# Commands
echo "This will be logged to the file"

gsutil cp gs://{bucket}/preprocess_gcs.py /home/huongn
gsutil cp gs://{bucket}/requirements_seg.txt /home/huongn
gsutil cp gs://{bucket}/utils.py /home/huongn

apt-get update && apt-get install -y \
    curl \
    git \
    ffmpeg \
    python3 \
    python3-venv \
    python3-pip \
    python3-setuptools \
    unzip \
    curl

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
python3 preprocess_gcs.py --bucket={bucket} --queue_id={queue_id} --tar_prefix={tar_prefix} --log_dir={log_dir} --seg_dir={seg_dir} --audio_dir={audio_dir} >> main.log 2>&1
"""


def main(
    iteration: Optional[int],
    bucket: str,
    project_id: str,
    service_account: str,
    vm_count: int,
    zone: str = "us-central1-c",
    machine_type: str = "n2d-highcpu-96",
    termination_action: str = "DELETE",
    base_name: str = "ow-download",
    queue_id: str = "ow-download",
    tar_prefix: str = "ow",
    log_dir: str = "logs",
    seg_dir: str = "segments",
    audio_dir: str = "audio",
    disk_size: int = 100,
):
    # Create startup script
    startup_script = STARTUP_SCRIPT.format(
        bucket=bucket,
        queue_id=queue_id,
        tar_prefix=tar_prefix,
        log_dir=log_dir,
        seg_dir=seg_dir,
        audio_dir=audio_dir,
    )

    with open("scripts/data/processing/gcs/startup_script.sh", "w") as f:
        f.write(startup_script)

    logger.info("Startup script created")

    # Upload download script and requirements file to GCS bucket
    command = [
        "gsutil",
        "cp",
        "scripts/data/processing/gcs/requirements_seg.txt",
        "scripts/data/processing/gcs/utils.py",
        "scripts/data/processing/gcs/preprocess_gcs.py",
        f"gs://{bucket}/",
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Error: Command failed with return code {result.returncode}")
        return None

    logger.info("Files uploaded to GCS bucket")

    logger.info("Creating spot instances")

    result = bulk_create_spot_instances(
        iteration=iteration,
        project_id=project_id,
        service_account=service_account,
        startup_script_path="scripts/data/processing/gcs/startup_script.sh",
        count=vm_count,
        zone=zone,
        machine_type=machine_type,
        termination_action=termination_action,
        base_name=base_name,
        disk_size=disk_size,
    )

    if result.stderr:
        logger.error(f"Error creating spot instances: {result.stderr}")
    else:
        logger.info("Spot instances created successfully")


if __name__ == "__main__":
    Fire(main)
    # main(
    #     iteration=None,
    #     bucket="ow-data-download",
    #     project_id="oe-training",
    #     service_account="349753783513-compute@developer.gserviceaccount.com",
    #     vm_count=2,
    #     zone="us-central1-c",
    #     machine_type="n2d-highcpu-96",
    #     termination_action="DELETE",
    #     base_name="ow-download",
    # )
