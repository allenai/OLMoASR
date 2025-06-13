set -ex

NUM_JOBS=43
OUTPUT_DIR="/weka/huongn/ow_seg_2"
SOURCE_DIR="/weka/huongn/ow_full"
LOG_DIR="/weka/huongn/metadata/ow_seg_2_logs"
MANIFEST_DIR="/weka/huongn/metadata/ow_seg_2_transcript_manifest"
AUDIO_ONLY=True
TRANSCRIPT_ONLY=False
MISSING_PAIR_DIR="/weka/huongn/ow_seg_2_missing_pairs"
JOB_BATCH_SIZE=10
REPLICAS=50
INIT_SHARD_IDX=0
IN_MEMORY=True

for ((i=37; i<NUM_JOBS; i++))
do
  START_SHARD_IDX=$((INIT_SHARD_IDX + (JOB_BATCH_SIZE * REPLICAS * i)))
  echo "START_SHARD_IDX=${START_SHARD_IDX}"
  echo "JOB_BATCH_SIZE=${JOB_BATCH_SIZE}"
  gantry run \
    --name "chunking_data_${i}" \
    --task-name "chunking_data_${i}" \
    --description "chunking audio-transcript pairs" \
    --allow-dirty \
    --preemptible \
    --beaker-image huongn/data_chunking \
    --workspace ai2/open-whisper \
    --cluster ai2/saturn-cirrascale \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/neptune-cirrascale \
    --cluster ai2/ceres-cirrascale \
    --budget ai2/oe-data \
    --replicas ${REPLICAS} \
    --weka oe-data-default:/weka \
    --priority low \
    --preemptible \
    -- /bin/bash -c "python scripts/data/processing/local/preprocess_flat.py \
      --output_dir=${OUTPUT_DIR} \
      --source_dir=${SOURCE_DIR} \
      --log_dir=${LOG_DIR} \
      --job_batch_size=${JOB_BATCH_SIZE} \
      --start_shard_idx=${START_SHARD_IDX} \
      --missing_pair_dir=${MISSING_PAIR_DIR} \
      --manifest_dir=${MANIFEST_DIR} \
      --audio_only=${AUDIO_ONLY} \
      --transcript_only=${TRANSCRIPT_ONLY} \
      --in_memory=${IN_MEMORY}
      " 
    echo "Sleeping for 10 seconds"
    sleep 10
done