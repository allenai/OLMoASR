BATCH_SIZE=64
NUM_WORKERS=14
SOURCE_FILE="/weka/huongn/intermediate_data/2K_valid_samples.jsonl.gz"
OUTPUT_DIR="/weka/huongn/intermediate_data/text_embds"
LEVEL="doc"
gantry run \
  --name "get_text_embds" \
  --description "get text embeddings" \
  --allow-dirty \
  --preemptible \
  --workspace ai2/open-whisper \
  --cluster ai2/saturn-cirrascale \
  --cluster ai2/jupiter-cirrascale-2 \
  --cluster ai2/neptune-cirrascale \
  --cluster ai2/ceres-cirrascale \
  --gpus 1 \
  --pip requirements/requirements-filter.txt \
  --beaker-image huongn/ow_vllm \
  --budget ai2/oe-data \
  --priority high \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/filtering/get_text_embds.py \
    --source_file=${SOURCE_FILE} \
    --output_dir=${OUTPUT_DIR} \
    --batch_size=${BATCH_SIZE} \
    --num_workers=${NUM_WORKERS} \
    --level=${LEVEL}"
      