gantry run \
  --name "unarchive_tars" \
  --description "unarchiving tars" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --workspace ai2/open-whisper \
  --cluster ai2/jupiter-cirrascale-2 \
  --cpus 40 \
  --budget ai2/oe-data \
  --replicas 50 \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/processing/unarchive_tar.py \
    unarchive_tar \
    --source_dir=/weka/huongn/1M_23K_tar \
    --base_output_dir=/weka/huongn/440K_full \
    --start_dir_idx=2699 \
    --batch_size=460 \
    --group_by=4
    "