EVAL_SETS=(
    "tedlium"
    "meanwhile"
    "rev16"
    "earnings21"
    "earnings22"
    "coraal"
    "kincaid46"
)

MODEL="whisper"
BATCH_SIZE=1
NUM_WORKERS=1
MODEL_SIZE="large"
LOG_DIR="logs"
CKPT="large.en.pt"
WANDB_LOG=True
WANDB_LOG_DIR="wandb"
EVAL_DIR="data/eval"
TASK="long_form_eval"

# Export environment variables
export WANDB_API_KEY="WANDB_API_KEY"
export HF_TOKEN="HF_TOKEN"

for eval_set in "${EVAL_SETS[@]}"; do
    echo "Running evaluation for ${eval_set}"
    
    # Run evaluation directly without gantry
    python scripts/eval/eval.py ${TASK} \
        --batch_size=${BATCH_SIZE} \
        --num_workers=${NUM_WORKERS} \
        --ckpt=${CKPT} \
        --eval_set=${eval_set} \
        --log_dir=${LOG_DIR} \
        --exp_name=None \
        --wandb_log=${WANDB_LOG} \
        --wandb_log_dir=${WANDB_LOG_DIR} \
        --eval_dir=${EVAL_DIR}
done