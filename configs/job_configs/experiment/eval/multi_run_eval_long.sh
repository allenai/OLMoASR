EVAL_SETS=(
    "tedlium"
    "meanwhile"
    "rev16"
    "earnings21"
    # "earnings22"
    # "coraal"
    "kincaid46"
)

MODEL="whisper"
BATCH_SIZE=1
NUM_WORKERS=1
MODEL_SIZE="large"
LOG_DIR="/results/huongn/ow_logs"
CKPT="/weka/huongn/whisper_ckpts/large-v3-turbo.pt"
WANDB_LOG=True
BOOTSTRAP=True
# WANDB_API_KEY="WANDB_API_KEY"
WANDB_API_KEY="HUONG_WANDB_API_KEY"
# WORKSPACE="ai2/open-whisper"
GITHUB_TOKEN="HUONG_GITHUB_TOKEN"
# GITHUB_TOKEN="GITHUB_TOKEN"
HF_TOKEN="HUONG_HF_TOKEN"
# HF_TOKEN="HF_TOKEN"
WORKSPACE="ai2/olmo3-webdata"
# WORKSPACE="ai2/open-whisper"
PRIORITY="high"
# PRIORITY="normal"

for eval_set in "${EVAL_SETS[@]}"; do
    echo "Running evaluation for ${eval_set}"
    gantry run \
        --name "ow_long_eval_${eval_set}_${MODEL}_${MODEL_SIZE}_en" \
        --task-name "ow_long_eval_${eval_set}_${MODEL}_${MODEL_SIZE}_en" \
        --description "Evaluation of model checkpoint" \
        --allow-dirty \
        --preemptible \
        --workspace ${WORKSPACE} \
        --cluster ai2/saturn-cirrascale \
        --cluster ai2/neptune-cirrascale \
        --cluster ai2/jupiter-cirrascale-2 \
        --cluster ai2/ceres-cirrascale \
        --gpus 1 \
        --beaker-image huongn/ow_eval \
        --pip requirements/requirements-eval.txt \
        --budget ai2/oe-data \
        --weka oe-data-default:/weka \
        --env-secret WANDB_API_KEY=${WANDB_API_KEY} \
        --gh-token-secret ${GITHUB_TOKEN} \
        --env-secret HF_TOKEN=${HF_TOKEN} \
        --priority ${PRIORITY} \
        -- /bin/bash -c "python scripts/eval/eval.py long_form_eval \
            --batch_size=${BATCH_SIZE} \
            --num_workers=${NUM_WORKERS} \
            --ckpt=${CKPT} \
            --eval_set=${eval_set} \
            --log_dir=${LOG_DIR} \
            --bootstrap=${BOOTSTRAP} \
            --exp_name=None \
            --wandb_log=${WANDB_LOG} \
            --wandb_log_dir=/results/huongn/wandb \
            --eval_dir=/weka/huongn/ow_eval \
            --hf_token=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"
done