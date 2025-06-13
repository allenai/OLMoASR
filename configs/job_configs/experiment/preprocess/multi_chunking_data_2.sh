#!/bin/bash

NUM_JOBS=12
OUTPUT_DIR="/weka/huongn/ow_seg_long"
SOURCE_DIR="/weka/huongn/ow_full"
LOG_DIR="/weka/huongn/metadata/ow_seg_long_logs"
MANIFEST_DIR="/weka/huongn/metadata/ow_seg_long_transcript_manifest"
AUDIO_ONLY=True
TRANSCRIPT_ONLY=False
MISSING_PAIR_DIR="/weka/huongn/ow_seg_long_missing_pairs"
JOB_BATCH_SIZE=50
REPLICAS=1
INIT_SHARD_IDX=550
IN_MEMORY=True

numbers=(
  550 700 750 1050 1300 4850 5350 6200 6850 7700
  8300 8400 9100 10250 10500 10650 10700 10750
  11600 13650 14400 14650 15200 15500 16200 16600
  17150 17500 18000 18500 18900 19350 19800 20150
  20300 20500 20600 20750 21050 21150
)

i=0
# for ((i=0; i<NUM_JOBS; i++)); do
for num in "${numbers[@]}"; do
    START_SHARD_IDX=$((num))
    ((i++))
    echo "START_SHARD_IDX=${START_SHARD_IDX}"
    YAML_PATH="/Users/huongn/Desktop/open_whisper/configs/job_configs/experiment/preprocess/multi_chunking_data.yaml"

    cat <<EOF > "$YAML_PATH"
version: v2
budget: ai2/oe-data
description: chunking audio-transcript pairs
retry:
  allowedTaskRetries: 5
tasks:
- name: chunking_data
  image:
    beaker: huongn/ow_data_processing
  command: ["python", "scripts/data/processing/local/preprocess_flat.py"]
  arguments:
    ["--output_dir=${OUTPUT_DIR}",
     "--source_dir=${SOURCE_DIR}",
     "--log_dir=${LOG_DIR}",
     "--job_batch_size=${JOB_BATCH_SIZE}",
     "--start_shard_idx=${START_SHARD_IDX}",
     "--missing_pair_dir=${MISSING_PAIR_DIR}",
     "--manifest_dir=${MANIFEST_DIR}",
     "--audio_only=${AUDIO_ONLY}",
     "--transcript_only=${TRANSCRIPT_ONLY}",
     "--in_memory=${IN_MEMORY}"]
  envVars:
  - name: PYTHONPATH
    value: /stage
  datasets:
  - mountPath: /weka
    source:
      weka: oe-data-default
  context:
    priority: high
    preemptible: false
  constraints:
    # cluster: [ai2/saturn-cirrascale]
    hostname: [jupiter-cs-aus-$((102 + i)).reviz.ai2.in]
  replicas: ${REPLICAS}
EOF

    beaker experiment create "$YAML_PATH"
done

# for ((i=0; i<NUM_JOBS; i++)); do
#     START_SHARD_IDX=$((INIT_SHARD_IDX + (JOB_BATCH_SIZE * REPLICAS * i)))
#     echo "START_SHARD_IDX=${START_SHARD_IDX}"
#     YAML_PATH="/Users/huongn/Desktop/open_whisper/configs/job_configs/experiment/preprocess/multi_chunking_data.yaml"

#     cat <<EOF > "$YAML_PATH"
# version: v2
# budget: ai2/oe-data
# description: chunking audio-transcript pairs
# retry:
#   allowedTaskRetries: 5
# tasks:
# - name: chunking_data
#   image:
#     beaker: huongn/ow_data_processing
#   command: ["python", "scripts/data/processing/local/preprocess_flat.py"]
#   arguments:
#     ["--output_dir=${OUTPUT_DIR}",
#      "--source_dir=${SOURCE_DIR}",
#      "--log_dir=${LOG_DIR}",
#      "--job_batch_size=${JOB_BATCH_SIZE}",
#      "--start_shard_idx=${START_SHARD_IDX}",
#      "--missing_pair_dir=${MISSING_PAIR_DIR}",
#      "--manifest_dir=${MANIFEST_DIR}",
#      "--audio_only=${AUDIO_ONLY}",
#      "--transcript_only=${TRANSCRIPT_ONLY}",
#      "--in_memory=${IN_MEMORY}"]
#   envVars:
#   - name: PYTHONPATH
#     value: /stage
#   datasets:
#   - mountPath: /weka
#     source:
#       weka: oe-data-default
#   context:
#     priority: high
#     preemptible: false
#   constraints:
#     # cluster: [ai2/saturn-cirrascale]
#     hostname: [neptune-cs-aus-$((256 + i)).reviz.ai2.in]
#   replicas: ${REPLICAS}
# EOF

#     beaker experiment create "$YAML_PATH"
# done