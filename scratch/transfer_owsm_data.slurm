#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24    # <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbjs-delta-cpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=gcs-transfer-owsm_data_v3.2
#SBATCH --time=6:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"
#SBATCH -e /work/hdd/bbjs/ai2_hngo1/slurm-%j.err
#SBATCH -o /work/hdd/bbjs/ai2_hngo1/slurm-%j.out
#SBATCH --mail-user=huongn@allenai.org
#SBATCH --mail-type="ALL"

# === CONFIGURATION ===
#!/bin/bash

SERVICE_ACCOUNT_KEY="/work/hdd/bbjs/ai2_hngo1/gcp_service_key.json"
LOCAL_DIR="/work/hdd/bbjs/shared/owsm_data/owsm_data_v3.2/train"
GCS_BUCKET="gs://ow-owsm-data/training_data"
GCLOUD_DIR="$HOME/google-cloud-sdk"
PROJECT_ID="oe-training"
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="temp"

conda activate "$ENV_NAME"

# === AUTHENTICATE GCLOUD ===
echo "Authenticating with service account..."
gcloud auth activate-service-account --key-file="$SERVICE_ACCOUNT_KEY"
gcloud config set project "$PROJECT_ID"

# === TRANSFER FILES TO GCS ===
echo "Starting file transfer from $LOCAL_DIR to $GCS_BUCKET"
gsutil -m cp -r "${LOCAL_DIR}" "$GCS_BUCKET"
echo "Transfer completed."