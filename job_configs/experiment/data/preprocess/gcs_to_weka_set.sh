gantry run \
  --name "gcs_to_weka_set" \
  --description "data transfer from gcs to weka" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --beaker-image huongn/gcs_to_weka \
  --workspace ai2/open-whisper \
  --cluster ai2/jupiter-cirrascale-2 \
  --cpus 10 \
  --pip requirements-filter.txt \
  --budget ai2/oe-data \
  --priority normal \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/data_transfer/download_gcs.py \
    download_files_set \
    --set_file=/weka/huongn/remaining_tars.txt \
    --local_dir=/weka/huongn/tars/4M \
    --bucket_name=ow-download-4m \
    --bucket_prefix=ow_4M_full \
    --service_account=349753783513-compute@developer.gserviceaccount.com \
    --key_file=/gcp_service_key.json \
    --log_file=/results/huongn/gcs_to_weka.log \
    " 