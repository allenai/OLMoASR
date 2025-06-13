JOB_BATCH_SIZE=10
BATCH_SIZE=64
NUM_WORKERS=28
REPLICAS=50
INIT_SHARD_IDX=0
SOURCE_DIR="/weka/huongn/intermediate_data/raw_full_jan25_tagged_sim_score_seg"
OUTPUT_DIR="/weka/huongn/intermediate_data/raw_full_jan25_tagged_all"
LEVEL="seg"
for ((i = 0; i < 43; i++))
do
  START_SHARD_IDX=$((INIT_SHARD_IDX + (JOB_BATCH_SIZE * REPLICAS * i)))
  echo "START_SHARD_IDX=${START_SHARD_IDX}"
  echo "JOB_BATCH_SIZE=${JOB_BATCH_SIZE}"
  gantry run \
  --name "tag_sim_score_seg" \
  --description "tag cosine sim score (seg)" \
  --allow-dirty \
  --preemptible \
  --workspace ai2/open-whisper \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 1 \
  --pip requirements/requirements-filter.txt \
  --beaker-image huongn/ow_filter \
  --budget ai2/oe-data \
  --replicas ${REPLICAS} \
  --priority normal \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/filtering/tag_sim_score.py \
    --source_dir=${SOURCE_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --start_shard_idx=${START_SHARD_IDX} \
    --job_batch_size=${JOB_BATCH_SIZE} \
    --batch_size=${BATCH_SIZE} \
    --num_workers=${NUM_WORKERS} \
    --level=${LEVEL}"
  echo "Sleeping for 10 seconds"
  sleep 10
done
      