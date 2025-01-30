EVAL_SETS=(
    "librispeech_clean"
    "librispeech_other"
    "artie_bias_corpus"
    "fleurs"
    "tedlium"
    "voxpopuli"
    "common_voice"
    "ami_ihm"
    "ami_sdm"
)

MODEL="ow"
BATCH_SIZE=96
MODEL_SIZE="tiny"
LOG_DIR="/results/huongn/ow_logs"
CKPT="/weka/huongn/ow_ckpts/mixed_no_repeat_comma_period_mach_gen_tiny_15e4_440K_bs32_ebs768_6workers_5pass_FULL_SHARD_011825_v3ecb7b3/eval_latesttrain_00349526_tiny_fsdp-train_grad-acc_bfloat16.pt"

for eval_set in "${EVAL_SETS[@]}"; do
    gantry run \
        --name "ow_eval_${eval_set}_${MODEL}_${MODEL_SIZE}_en" \
        --task-name "ow_eval_${eval_set}_${MODEL}_${MODEL_SIZE}_en" \
        --description "Evaluation of model checkpoint" \
        --allow-dirty \
        --no-nfs \
        --preemptible \
        --workspace ai2/open-whisper \
        --cluster ai2/ceres-cirrascale \
        --gpus 1 \
        --beaker-image huongn/ow_eval \
        --pip requirements-eval.txt \
        --budget ai2/oe-data \
        --weka oe-data-default:/weka \
        --env-secret WANDB_API_KEY=WANDB_API_KEY \
        --priority normal \
        -- /bin/bash -c "python scripts/eval/eval.py main \
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