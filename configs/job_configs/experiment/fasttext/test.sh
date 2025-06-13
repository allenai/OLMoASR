EVAL_SET="common_voice"
EVAL_TRAIN_DIR="/weka/huongn/ow_eval_train"
MODEL_DIR="/weka/huongn/ow_eval_fasttext"
OUTPUT_DIR="/weka/huongn/ow_eval_fasttext"
WORKSPACE="ai2/open-whisper" #ai2/open-whisper ai2/vida
HF_TOKEN="HF_TOKEN" #HUONGN_HF_TOKEN HF_TOKEN
GITHUB_TOKEN="GITHUB_TOKEN" #HUONGN_GITHUB_TOKEN GITHUB_TOKEN
PRIORITY="high"

gantry run \
    --name "test_fasttext_${EVAL_SET}" \
    --task-name "test_fasttext_${EVAL_SET}" \
    --description "Test fasttext model for ${EVAL_SET}" \
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
    -- /bin/bash -c "/stage/fastText-0.9.2/fasttext predict-prob ${MODEL_DIR}/${EVAL_SET}.bin ${EVAL_TRAIN_DIR}/${EVAL_SET}.test 2 > ${OUTPUT_DIR}/${EVAL_SET}_pred.txt"
#     -- /bin/bash -c "/stage/fastText-0.9.2/fasttext predict-prob ${MODEL_DIR}/${EVAL_SET}.bin ${EVAL_TRAIN_DIR}/${EVAL_SET}.test 2 > ${OUTPUT_DIR}/${EVAL_SET}_pred.txt"
#    -- /bin/bash -c "/stage/fastText-0.9.2/fasttext test ${MODEL_DIR}/${EVAL_SET}.bin ${EVAL_TRAIN_DIR}/${EVAL_SET}.test"