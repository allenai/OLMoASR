BATCH_SIZE=80
REPLICAS=10
INIT_DIR_IDX=26000
for ((i = 0; i < 32; i++))
do
  START_DIR_IDX=$((INIT_DIR_IDX + (BATCH_SIZE * REPLICAS * i)))
  echo "START_DIR_IDX=${START_DIR_IDX}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  gantry run \
    --name "gcs_to_weka" \
    --description "data transfer from gcs to weka" \
    --allow-dirty \
    --no-nfs \
    --preemptible \
    --beaker-image huongn/gcs_to_weka \
    --workspace ai2/open-whisper \
    --cluster ai2/neptune-cirrascale \
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
      --local_dir=/weka/huongn/tars/8M/seg_250_10K \
      --bucket_name=ow-download-4m \
      --bucket_prefix=segments \
      --service_account=349753783513-compute@developer.gserviceaccount.com \
      --key_file=/gcp_service_key.json \
      --log_file=/results/huongn/gcs_to_weka.log \
      --padding=8 \
      --file_ext=tar.gz
      "
  # echo "Sleeping for 10 minutes"
  # sleep 600
    echo "Sleeping for 8 minutes"
    sleep 480
done