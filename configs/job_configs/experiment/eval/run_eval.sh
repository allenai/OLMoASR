MODEL="whisper"
EVAL_SET="tedlium"
BATCH_SIZE=96
MODEL_SIZE="tiny"
LOG_DIR="/results/huongn/ow_logs"
CKPT="/weka/huongn/whisper_ckpts/tiny.en.pt"
WANDB_LOG=True
BOOTSTRAP=False
TASK="long_form_eval" # or "short_form_eval"

gantry run \
    --name "ow_eval_${EVAL_SET}_${MODEL}_${MODEL_SIZE}_en" \
    --task-name "ow_eval_${EVAL_SET}_${MODEL}_${MODEL_SIZE}_en" \
    --description "Evaluation of model checkpoint" \
    --allow-dirty \
    --preemptible \
    --workspace ai2/open-whisper \
    --cluster ai2/neptune-cirrascale \
    --gpus 1 \
    --beaker-image huongn/ow_eval \
    --pip requirements/requirements-eval.txt \
    --budget ai2/oe-data \
    --weka oe-data-default:/weka \
    --env-secret WANDB_API_KEY=WANDB_API_KEY \
    --priority high \
    -- /bin/bash -c "python scripts/eval/eval.py ${TASK} \
        --batch_size=${BATCH_SIZE} \
        --num_workers=8 \
        --ckpt=${CKPT} \
        --eval_set=${EVAL_SET} \
        --log_dir=${LOG_DIR} \
        --wandb_log=${WANDB_LOG} \
        --wandb_log_dir=/results/huongn/wandb \
        --eval_dir=/weka/huongn/ow_eval \
        --hf_token=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" \
        # --cuda=True \
        # --bootstrap=${BOOTSTRAP}"