DATA_DIR="/weka/huongn/ow_full"
OUTPUT_DIR="/weka/huongn/ow_full_jsonl"
BATCH_SIZE=25
INIT_SHARD_IDX=0
REPLICAS=20
BUCKET="allennlp-mattj"
BUCKET_PREFIX="openwhisper/pretraining_data/jsonl_no_mach"
WORKSPACE="ai2/open-whisper" #ai2/open-whisper ai2/vida
HF_TOKEN="HF_TOKEN" #HUONGN_HF_TOKEN HF_TOKEN
GITHUB_TOKEN="GITHUB_TOKEN" #HUONGN_GITHUB_TOKEN GITHUB_TOKEN
PRIORITY="high"

for ((i = 29; i < 43; i++))
do
    START_SHARD_IDX=$((INIT_SHARD_IDX + (BATCH_SIZE * REPLICAS * i)))
    END_SHARD_IDX=$((START_SHARD_IDX + (BATCH_SIZE * REPLICAS)))
    echo "START_SHARD_IDX=${START_SHARD_IDX}"
    gantry run \
        --name "text_to_jsonl" \
        --description "Generate JSONL format training data" \
        --allow-dirty \
        --no-nfs \
        --beaker-image huongn/ow_gen_jsonl \
        --workspace ${WORKSPACE} \
        --cluster ai2/neptune-cirrascale \
        --cpus 62 \
        --pip requirements-filter.txt \
        --budget ai2/oe-data \
        --replicas 20 \
        --priority ${PRIORITY} \
        --env-secret HF_TOKEN=${HF_TOKEN} \
        --gh-token-secret ${GITHUB_TOKEN} \
        --weka oe-data-default:/weka \
        -- /bin/bash -c "python scripts/data/processing/text_to_jsonl.py \
            --data_dir=${DATA_DIR} \
            --output_dir=${OUTPUT_DIR} \
            --batch_size=${BATCH_SIZE} \
            --start_shard_idx=${START_SHARD_IDX} \
            --end_shard_idx=${END_SHARD_IDX} \
            --bucket=${BUCKET} \
            --bucket_prefix=${BUCKET_PREFIX}
            "
    echo "Sleeping for 2 minutes"
    sleep 120
done