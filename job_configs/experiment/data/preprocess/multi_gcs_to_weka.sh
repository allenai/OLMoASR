BATCH_SIZE=40
REPLICAS=10
for ((i = 0; i < 31; i++))
do
  START_DIR_IDX=$((3600 + (BATCH_SIZE * REPLICAS * i)))
  gantry run \
    --name "gcs_to_weka" \
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
    --replicas ${REPLICAS} \
    --priority normal \
    --weka oe-data-default:/weka \
    -- /bin/bash -c "python scripts/data/data_transfer/download_gcs.py \
      download_files_batch \
      --start_dir_idx=${START_DIR_IDX} \
      --batch_size=${BATCH_SIZE} \
      --local_dir=/weka/huongn/tars/4M/seg_250 \
      --bucket_name=ow-download-4m \
      --bucket_prefix=segments \
      --service_account=349753783513-compute@developer.gserviceaccount.com \
      --key_file=/gcp_service_key.json \
      --log_file=/results/huongn/gcs_to_weka.log \
      --padding=8 \
      --file_ext=tar.gz
      "
  sleep 600
done