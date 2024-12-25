set -ex

OUTPUT_DIR="/weka/huongn/leftover_seg"
SOURCE_DIR="/weka/huongn/leftover"
LOG_DIR="/results/huongn"
PREPROC_FAIL_DIR="/weka/huongn/leftover_seg_fail"
MISSING_PAIR_DIR="/weka/huongn/leftover_missing_pairs"
JOBS_BATCH_SIZE=13
START_SHARD_IDX=0
END_SHARD_IDX=12
IN_MEMORY=True
JOB_BATCH_IDX=0

gantry run \
  --name "chunking_data_313" \
  --description "chunking audio-transcript pairs" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --beaker-image huongn/data_chunking \
  --workspace ai2/open-whisper \
  --cluster ai2/jupiter-cirrascale-2 \
  --cpus 4.5 \
  --budget ai2/oe-data \
  --replicas ${JOBS_BATCH_SIZE} \
  --weka oe-data-default:/weka \
  --env JOB_BATCH_IDX=${JOB_BATCH_IDX} \
  -- /bin/bash -c "python scripts/data/processing/preprocess_flat.py \
    --output_dir=${OUTPUT_DIR} \
    --source_dir=${SOURCE_DIR} \
    --log_dir=${LOG_DIR} \
    --preproc_fail_dir=${PREPROC_FAIL_DIR} \
    --missing_pair_dir=${MISSING_PAIR_DIR} \
    --jobs_batch_size=${JOBS_BATCH_SIZE} \
    --start_shard_idx=${START_SHARD_IDX} \
    --end_shard_idx=${END_SHARD_IDX} \
    --in_memory=${IN_MEMORY}
    " 