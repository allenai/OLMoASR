#!/bin/bash

DIRS=(
  # "FLEURS"
  # "GigaST"
  # "GigaSpeech"
  # "LibriSpeech"
  # "MLS"
  # "MuST-C"
  # "TEDLIUM3_filtered"
  # "VoxPopuli"
  # "ami"
  # "commonvoice"
  # "fisher_callhome"
  # "openslr"
  # "swbd"
  # "training_data"
  # "vctk"
  # "voxforge"
  # "wsj"
  "SPGISpeech"
)
LOCAL_DIR="/weka/huongn/training_data/owsm_data"
BUCKET_NAME="ow-owsm-data"
for i in "${!DIRS[@]}"; do
  dir=${DIRS[$i]}
  echo "Processing directory: ${dir}"
  gantry run \
    --name "gcs_to_weka_${dir}" \
    --description "data transfer from gcs to weka" \
    --allow-dirty \
    --preemptible \
    --beaker-image huongn/data_transfer \
    --workspace ai2/open-whisper \
    --hostname saturn-cs-aus-$((242 + i)).reviz.ai2.in \
    --pip requirements/requirements-data.txt \
    --budget ai2/oe-data \
    --priority normal \
    --weka oe-data-default:/weka \
    -- /bin/bash -c "python scripts/data/data_transfer/file_transfer_gcs.py download_dir \
      --local_dir=${LOCAL_DIR}/${dir} \
      --bucket_name=${BUCKET_NAME} \
      --bucket_prefix=${dir}/train \
      --service_account=349753783513-compute@developer.gserviceaccount.com \
      --key_file=/gcp_service_key.json \
      --log_file=/results/huongn/gcs_to_weka.log
      "
  echo "Sleeping for 3 mins"
  sleep 180
done