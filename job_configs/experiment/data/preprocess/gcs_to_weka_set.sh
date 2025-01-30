gantry run \
  --name "gcs_to_weka_set" \
  --description "data transfer from gcs to weka" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --beaker-image huongn/gcs_to_weka \
  --workspace ai2/open-whisper \
  --cluster ai2/neptune-cirrascale \
  --cpus 10 \
  --replicas 4 \
  --pip requirements-filter.txt \
  --budget ai2/oe-data \
  --priority normal \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/data_transfer/download_gcs.py \
    download_files_set \
    --set_file=/weka/huongn/metadata/seg_tars_ml_2.txt \
    --local_dir=/weka/huongn/tars/ml/seg_rnd2 \
    --bucket_name=ow-download-ml \
    --bucket_prefix=segments_rnd2 \
    --service_account=349753783513-compute@developer.gserviceaccount.com \
    --key_file=/gcp_service_key.json \
    --log_file=/results/huongn/gcs_to_weka.log \
    --batches=4
    " 