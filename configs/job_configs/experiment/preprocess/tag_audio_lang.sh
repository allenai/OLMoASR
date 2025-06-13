JOB_BATCH_SIZE=5
BATCH_SIZE=64
NUM_WORKERS=12
REPLICAS=50
INIT_SHARD_IDX=0
SOURCE_DIR="/weka/huongn/data/unfiltered/all_seg_jan_25"
OUTPUT_DIR="/weka/huongn/data/audio_lang"
for ((i = 49; i < 85; i++))
do
  START_SHARD_IDX=$((INIT_SHARD_IDX + (JOB_BATCH_SIZE * REPLICAS * i)))
  echo "START_SHARD_IDX=${START_SHARD_IDX}"
  echo "JOB_BATCH_SIZE=${JOB_BATCH_SIZE}"
  gantry run \
  --name "tag_audio_lang" \
  --description "tag audio language" \
  --allow-dirty \
  --preemptible \
  --workspace ai2/open-whisper \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 1 \
  --pip requirements/requirements-filter.txt \
  --budget ai2/oe-data \
  --replicas ${REPLICAS} \
  --priority normal \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/filtering/tag_audio_lang.py \
    --source_dir=${SOURCE_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --start_shard_idx=${START_SHARD_IDX} \
    --job_batch_size=${JOB_BATCH_SIZE} \
    --batch_size=${BATCH_SIZE} \
    --num_workers=${NUM_WORKERS}"
  echo "Sleeping for 60 seconds"
  sleep 60
done
      