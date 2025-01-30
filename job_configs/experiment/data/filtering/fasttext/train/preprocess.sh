EVAL_SET="common_voice"
EVAL_TRAIN_DIR="/weka/huongn/ow_eval_train"
TRAIN_DIR=/weka/huongn/samples_dicts/filtered/v2_simplefilter # "/weka/huongn/ow_full"
WORKSPACE="ai2/open-whisper" #ai2/open-whisper ai2/vida
HF_TOKEN="HF_TOKEN" #HUONGN_HF_TOKEN HF_TOKEN #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
GITHUB_TOKEN="GITHUB_TOKEN" #HUONGN_GITHUB_TOKEN GITHUB_TOKEN
PRIORITY="high"

gantry run \
    --name "gen_data_fasttext_${EVAL_SET}" \
    --task-name "gen_data_fasttext_${EVAL_SET}" \
    --description "Generate data to train fasttext model for ${EVAL_SET}" \
    --allow-dirty \
    --no-nfs \
    --workspace ${WORKSPACE} \
    --cluster ai2/neptune-cirrascale \
    --gpus 1 \
    --shared-memory 20GiB \
    --pip requirements-filter.txt \
    --beaker-image huongn/ow_filter \
    --budget ai2/oe-data \
    --weka oe-data-default:/weka \
    --priority ${PRIORITY} \
    --env-secret HF_TOKEN=${HF_TOKEN} \
    --gh-token-secret ${GITHUB_TOKEN} \
    -- /bin/bash -c "python scripts/data/filtering/fasttext/preprocess.py \
        --eval_set=${EVAL_SET} \
        --eval_train_dir=${EVAL_TRAIN_DIR} \
        --train_dir=${TRAIN_DIR} \
        --segment_filter=True \
        --jsonl_input=False"