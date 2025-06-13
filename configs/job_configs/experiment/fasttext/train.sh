EVAL_SET="librispeech_other"
EVAL_TRAIN_DIR="/weka/huongn/ow_eval_train"
MODEL_DIR="/weka/huongn/ow_eval_fasttext"
WORKSPACE="ai2/open-whisper" #ai2/open-whisper ai2/vida
HF_TOKEN="HF_TOKEN" #HUONGN_HF_TOKEN HF_TOKEN
GITHUB_TOKEN="GITHUB_TOKEN" #HUONGN_GITHUB_TOKEN GITHUB_TOKEN
PRIORITY="high"

gantry run \
    --name "train_fasttext_${EVAL_SET}" \
    --task-name "train_fasttext_${EVAL_SET}" \
    --description "Train fasttext model for ${EVAL_SET}" \
    --allow-dirty \
    --no-nfs \
    --workspace ${WORKSPACE} \
    --cluster ai2/neptune-cirrascale \
    --gpus 1 \
    --shared-memory 20GiB \
    --pip requirements/requirements-filter.txt \
    --beaker-image huongn/ow_filter \
    --budget ai2/oe-data \
    --weka oe-data-default:/weka \
    --priority ${PRIORITY} \
    --env-secret HF_TOKEN=${HF_TOKEN} \
    --gh-token-secret ${GITHUB_TOKEN} \
    -- /bin/bash -c "/stage/fastText-0.9.2/fasttext supervised -input ${EVAL_TRAIN_DIR}/${EVAL_SET}.train -output ${MODEL_DIR}/${EVAL_SET} -lr 0.01 -epoch 10"