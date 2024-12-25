gantry run \
  --name "unarchive_tars" \
  --description "unarchiving tars" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --workspace ai2/open-whisper \
  --cluster ai2/neptune-cirrascale \
  --cpus 20 \
  --budget ai2/oe-data \
  --replicas 50 \
  --priority normal \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/processing/unarchive_tar.py \
    unarchive_tar \
    --source_dir=/weka/huongn/tars/8M/full_250_10K \
    --base_output_dir=/weka/huongn/ow_full \
    --start_dir_idx=12449 \
    --start_tar_idx=0 \
    --end_tar_idx=9999 \
    --batch_size=200 \
    --group_by=4
    "