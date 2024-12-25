EVAL_SETS=(
    # "librispeech_clean"
    # "librispeech_other"
    # "artie_bias_corpus"
    # "fleurs"
    # "tedlium"
    "voxpopuli"
    "common_voice"
    # "ami_ihm"
    # "ami_sdm"
)

MODEL="whisper"
BATCH_SIZE=80
MODEL_SIZE="tiny"
LOG_DIR="/results/huongn/ow_logs"
CKPT="/weka/huongn/whisper_ckpts/tiny.en.pt"

for eval_set in "${EVAL_SETS[@]}"; do
    if [[ "$eval_set" == "common_voice" ]]; then
        batch_size=80
    else
        batch_size=96
    fi

    gantry run \
        --name "ow_eval_${eval_set}_${MODEL}_${MODEL_SIZE}_en" \
        --task-name "ow_eval_${eval_set}_${MODEL}_${MODEL_SIZE}_en" \
        --description "Evaluation of model checkpoint" \
        --allow-dirty \
        --no-nfs \
        --preemptible \
        --workspace ai2/open-whisper \
        --cluster ai2/neptune-cirrascale \
        --gpus 1 \
        --beaker-image huongn/ow_eval \
        --pip requirements-eval.txt \
        --budget ai2/oe-data \
        --weka oe-data-default:/weka \
        --env-secret WANDB_API_KEY=WANDB_API_KEY \
        --priority normal \
        -- /bin/bash -c "python scripts/eval/eval.py \
            --batch_size=${BATCH_SIZE} \
            --num_workers=8 \
            --ckpt=${CKPT} \
            --eval_set=${eval_set} \
            --log_dir=${LOG_DIR} \
            --wandb_log=True \
            --wandb_log_dir=/results/huongn/wandb \
            --eval_dir=/weka/huongn/ow_eval \
            --hf_token=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"
done