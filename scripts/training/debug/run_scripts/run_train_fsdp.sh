set -ex

MODEL_SIZE="large"
LEARNING_RATE=1.5e-3
LR="15e4"
AUDIO_HOURS=200
BATCH_SIZE=16
GPU_COUNT=8
REPLICAS=4
ACCUMULATION_STEPS=4
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GPU_COUNT * REPLICAS * ACCUMULATION_STEPS))
EPOCHS=1
NUM_WORKERS=6
DATE="012725"
EXP_NAME="${MODEL_SIZE}_${LR}_${AUDIO_HOURS}_bs${BATCH_SIZE}_ebs${EFFECTIVE_BATCH_SIZE}_${GPU_COUNT}gpus_${NUM_WORKERS}workers_${EPOCHS}pass_DDP_${DATE}"

JOB_TYPE="debug"
SAMPLES_DICTS_DIR="/weka/huongn/samples_dicts/debug"
EPOCH_STEPS=85
TRAIN_STEPS=340
PREFETCH_FACTOR=2
PROFILE=True

SCRIPT="scripts/training/debug/train_ddp.py"

torchrun --nnodes ${REPLICAS}:${REPLICAS} --nproc_per_node ${GPU_COUNT} ${SCRIPT} \
    --model_variant=${MODEL_SIZE} \
    --exp_name=${EXP_NAME} \
    --job_type=${JOB_TYPE} \
    --samples_dicts_dir=${SAMPLES_DICTS_DIR} \
    --epoch_steps=${EPOCH_STEPS} \
    --train_steps=${TRAIN_STEPS} \
    --lr=${LEARNING_RATE} \
    --betas='(0.9, 0.98)' \
    --eps=1e-6 \
    --weight_decay=0.1 \
    --max_grad_norm=1.0 \
    --eff_batch_size=${EFFECTIVE_BATCH_SIZE} \
    --train_batch_size=${BATCH_SIZE} \
    --num_workers=${NUM_WORKERS} \
    --pin_memory=True \
    --shuffle=True \
    --prefetch_factor=${PREFETCH_FACTOR} \
    --persistent_workers=True \
    --verbose=False \
    --precision="float16" \
    --log_dir=/stage