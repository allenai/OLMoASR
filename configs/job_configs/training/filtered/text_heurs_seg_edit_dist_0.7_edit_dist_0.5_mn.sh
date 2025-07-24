set -ex

FILTER="text_heurs_seg_edit_dist_0.7_edit_dist_0.5_long" # <-- can be changed (threshold filter)
MODEL_SIZE="medium"
LEARNING_RATE=1.5e-3 # <-- might need to tune
LR="15e4" # <-- might need to tune
AUDIO_HOURS=440
BATCH_SIZE=16
GPU_COUNT=8
REPLICAS=2
ACCUMULATION_STEPS=2
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GPU_COUNT * REPLICAS * ACCUMULATION_STEPS))
EPOCHS=5
NUM_WORKERS=16 # <-- might need to tune
SHARDING_STRATEGY="FULL_SHARD"
EXP_NAME="${FILTER}_${MODEL_SIZE}_${LR}_${AUDIO_HOURS}K_bs${BATCH_SIZE}_ebs${EFFECTIVE_BATCH_SIZE}_${NUM_WORKERS}workers_${EPOCHS}pass"
echo ${EFFECTIVE_BATCH_SIZE}

# Local paths for training
HARDWARE="H100" # <-- can be changed
JOB_TYPE="tuning"
SAMPLES_DICTS_DIR="data/training/filtered/text_heurs_seg_edit_dist_0.7_edit_dist_0.5/220K" # <-- can be changed (threshold filter)
EPOCH_STEPS=51577 # <-- change depending on dataset
TRAIN_STEPS=524288 # <-- change depending on effective batch size
EVAL_BATCH_SIZE=8 # <-- change depending on EVAL ON GPU / CPU
RUN_EVAL=False
TRAIN_LOG_FREQ=1000
EVAL_FREQ=5000
CKPT_FREQ=1000

VERBOSE=False
PRECISION="bfloat16"

ASYNC_EVAL=True
EVAL_SCRIPT_PATH="scripts/eval/eval.py"
EVAL_WANDB_LOG=True
EVAL_ON_GPU=True # <-- can be changed

# Export environment variables
export WANDB_API_KEY="WANDB_API_KEY"
export HF_TOKEN="HF_TOKEN"
export WANDB_DIR="logs"
export NCCL_DEBUG=INFO

SCRIPT="scripts/training/prod/train_fsdp_timestamps.py"

# async eval switch
if [ "$ASYNC_EVAL" == "True" ]; then
    EXP_NAME="${EXP_NAME}_AsyncEval"
fi

# eval on gpu/cpu switch
if [ "$EVAL_ON_GPU" == "True" ] && [ "$ASYNC_EVAL" == "True" ]; then
    EXP_NAME="${EXP_NAME}_EvalOnGPU"
elif [ "$EVAL_ON_GPU" == "False" ] && [ "$ASYNC_EVAL" == "True" ]; then
    EXP_NAME="${EXP_NAME}_EvalOnCPU"
fi

EXP_NAME="${EXP_NAME}_evalbs${EVAL_BATCH_SIZE}"

DATE=$(date +"%m%d%y")
EXP_NAME="${EXP_NAME}_${DATE}"

echo "Final EXP_NAME: $EXP_NAME"

# Run FSDP training directly without gantry
torchrun --nnodes ${REPLICAS}:${REPLICAS} --nproc_per_node ${GPU_COUNT} --rdzv_id=101 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 ${SCRIPT} \
    --model_variant=${MODEL_SIZE} \
    --exp_name=${EXP_NAME} \
    --job_type=${JOB_TYPE} \
    --samples_dicts_dir=${SAMPLES_DICTS_DIR} \
    --train_steps=${TRAIN_STEPS} \
    --epoch_steps=${EPOCH_STEPS} \
    --ckpt_file_name=None \
    --ckpt_dir=checkpoints \
    --log_dir=logs \
    --eval_dir=data/eval \
    --run_id_dir=run_ids \
    --lr=${LEARNING_RATE} \
    --betas='(0.9, 0.98)' \
    --eps=1e-6 \
    --weight_decay=0.1 \
    --max_grad_norm=1.0 \
    --eff_batch_size=${EFFECTIVE_BATCH_SIZE} \
    --train_batch_size=${BATCH_SIZE} \
    --eval_batch_size=${EVAL_BATCH_SIZE} \
    --num_workers=${NUM_WORKERS} \
    --prefetch_factor=2 \
    --pin_memory=True \
    --shuffle=True \
    --persistent_workers=True \
    --run_eval=${RUN_EVAL} \
    --train_log_freq=${TRAIN_LOG_FREQ} \
    --eval_freq=${EVAL_FREQ} \
    --ckpt_freq=${CKPT_FREQ} \
    --verbose=${VERBOSE} \
    --precision=${PRECISION} \
    --hardware=${HARDWARE} \
    --eval_script_path=${EVAL_SCRIPT_PATH} \
    --eval_wandb_log=${EVAL_WANDB_LOG} \
    --eval_on_gpu=${EVAL_ON_GPU} \
    --sharding_strategy=${SHARDING_STRATEGY}