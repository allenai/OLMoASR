set -ex

SOURCE_DIR="/weka/huongn/440K_full"
LOG_DIR="/weka/huongn"
START_SHARD_IDX=7744
END_SHARD_IDX=8448
BATCH_SIZE=704
NUM_REPLICAS=1

gantry run \
  --name "filter_en_only" \
  --description "Filtering for English transcripts only" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --pip requirements-filter.txt \
  --workspace ai2/open-whisper \
  --cluster ai2/neptune-cirrascale \
  --cpus 10 \
  --priority normal \
  --budget ai2/oe-data \
  --replicas ${NUM_REPLICAS} \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/filtering/filter_en_only.py \
    --source_dir=${SOURCE_DIR} \
    --log_dir=${LOG_DIR} \
    --start_shard_idx=${START_SHARD_IDX} \
    --end_shard_idx=${END_SHARD_IDX} \
    --batch_size=${BATCH_SIZE} \
    "