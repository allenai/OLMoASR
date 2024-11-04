set -ex

SOURCE_DIR="/weka/huongn/440K_seg"
LOG_DIR="/results/huongn"
START_SHARD_IDX=2449
END_SHARD_IDX=8448
BATCH_SIZE=60

gantry run \
  --name "modify_silent_vtt" \
  --description "modify silent (empty) VTT transcripts" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --workspace ai2/open-whisper \
  --cluster ai2/neptune-cirrascale \
  --cpus 10 \
  --budget ai2/oe-data \
  --replicas 100 \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/processing/modify_silent_vtt.py \
    --source_dir=${SOURCE_DIR} \
    --log_dir=${LOG_DIR} \
    --start_shard_idx=${START_SHARD_IDX} \
    --end_shard_idx=${END_SHARD_IDX} \
    --batch_size=${BATCH_SIZE} \
    "