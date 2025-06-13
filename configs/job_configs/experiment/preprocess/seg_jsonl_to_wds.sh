WORKSPACE="ai2/open-whisper" #ai2/open-whisper ai2/vida
HF_TOKEN="HF_TOKEN" #HUONGN_HF_TOKEN HF_TOKEN
GITHUB_TOKEN="GITHUB_TOKEN" #HUONGN_GITHUB_TOKEN GITHUB_TOKEN
PRIORITY="normal"
INPUT_DIR="/weka/huongn/samples_dicts/filtered/text_heurs_1_jan25"
OUTPUT_DIR="/weka/huongn/ow_wds/filtered/text_heurs_1_jan25"
SUBSAMPLED_SIZE=2500
LOG_DIR="/results/huongn"
SPLIT_FACTOR=2
REPLICAS=15
BATCH_SIZE=1413 # 1413 * 15 = 21195

gantry run \
    --name "seg_jsonl_to_wds" \
    --description "Generate WDS format of data from segmented JSONLs" \
    --allow-dirty \
    --no-nfs \
    --beaker-image huongn/ow_gen_jsonl \
    --workspace ${WORKSPACE} \
    --cluster ai2/neptune-cirrascale \
    --cpus 62 \
    --replicas ${REPLICAS} \
    --pip requirements-filter.txt \
    --budget ai2/oe-data \
    --priority ${PRIORITY} \
    --env-secret HF_TOKEN=${HF_TOKEN} \
    --gh-token-secret ${GITHUB_TOKEN} \
    --weka oe-data-default:/weka \
    -- /bin/bash -c "python scripts/data/processing/seg_jsonl_to_wds.py \
        --input_dir=${INPUT_DIR} \
        --output_dir=${OUTPUT_DIR} \
        --subsampled_size=${SUBSAMPLED_SIZE} \
        --log_dir=${LOG_DIR} \
        --split_factor=${SPLIT_FACTOR} \
        --batch_size=${BATCH_SIZE}"