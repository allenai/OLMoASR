set -ex

SOURCE_DIR="/weka/huongn/440K_seg"
LOG_DIR="/weka/huongn/check_even_seg"
START_SHARD_IDX=2449
END_SHARD_IDX=8448
BATCH_SIZE=60
# transcript ext here is VTT

gantry run \
  --name "check_even_seg" \
  --description "check if segmentation resulted in even audio and transcript files" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --workspace ai2/open-whisper \
  --cluster ai2/jupiter-cirrascale-2 \
  --cpus 10 \
  --budget ai2/oe-data \
  --replicas 100 \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/processing/check_even_seg.py \
    --source_dir=${SOURCE_DIR} \
    --log_dir=${LOG_DIR} \
    --start_shard_idx=${START_SHARD_IDX} \
    --end_shard_idx=${END_SHARD_IDX} \
    --batch_size=${BATCH_SIZE} \
    "