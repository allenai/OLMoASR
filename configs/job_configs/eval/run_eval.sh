MODEL="whisper"
EVAL_SET="tedlium" # or any other eval set
BATCH_SIZE=96
NUM_WORKERS=8
MODEL_SIZE="tiny"
LOG_DIR="logs"
CKPT="tiny.en.pt"
WANDB_LOG=True
WANDB_LOG_DIR="wandb"
EVAL_DIR="data/eval"
TASK="short_form_eval" # or "long_form_eval"

export HF_TOKEN="HF_TOKEN"
export WANDB_API_KEY="WANDB_API_KEY"

# Run evaluation directly without gantry
python scripts/eval/eval.py ${TASK} \
    --batch_size=${BATCH_SIZE} \
    --num_workers=${NUM_WORKERS} \
    --ckpt=${CKPT} \
    --eval_set=${EVAL_SET} \
    --log_dir=${LOG_DIR} \
    --wandb_log=${WANDB_LOG} \
    --wandb_log_dir=${WANDB_LOG_DIR} \
    --eval_dir=${EVAL_DIR}"
