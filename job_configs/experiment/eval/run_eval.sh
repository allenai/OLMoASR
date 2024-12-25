MODEL="whisper"
EVAL_SET="librispeech_other"
BATCH_SIZE=80
MODEL_SIZE="tiny"
LOG_DIR="/results/huongn/ow_logs"
CKPT="/weka/huongn/whisper_ckpts/tiny.en.pt"

gantry run \
    --name "ow_eval_${EVAL_SET}_${MODEL}_${MODEL_SIZE}_en" \
    --task-name "ow_eval_${EVAL_SET}_${MODEL}_${MODEL_SIZE}_en" \
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
        --eval_set=${EVAL_SET} \
        --log_dir=${LOG_DIR} \
        --wandb_log=True \
        --wandb_log_dir=/results/huongn/wandb \
        --eval_dir=/weka/huongn/ow_eval \
        --hf_token=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"