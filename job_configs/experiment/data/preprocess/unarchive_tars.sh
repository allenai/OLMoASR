gantry run \
  --name "unarchive_tars" \
  --description "unarchiving tars" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --workspace ai2/open-whisper \
  --cluster ai2/ceres-cirrascale \
  --cpus 10 \
  --budget ai2/oe-data \
  --replicas 18 \
  --priority normal \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/processing/unarchive_tar.py \
    unarchive_tar \
    --source_dir=/weka/huongn/tars/8M/full_250_10K \
    --base_output_dir=/weka/huongn/ow_full \
    --start_dir_idx=14949 \
    --start_tar_idx=10000 \
    --end_tar_idx=34983 \
    --batch_size=1388 \
    --group_by=4
    "