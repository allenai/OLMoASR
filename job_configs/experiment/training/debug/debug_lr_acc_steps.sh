set -ex

MODEL_SIZE="tiny"
LEARNING_RATE=1.5e-3
LR="15e4"
BATCH_SIZE=16
EFFECTIVE_BATCH_SIZE=256
DATE="122024"
EXP_NAME="debug_${MODEL_SIZE}_${LR}_bs${BATCH_SIZE}_ebs${EFFECTIVE_BATCH_SIZE}_${DATE}"

GPU_COUNT=8

JOB_TYPE="debug"
SAMPLES_DICTS_DIR="/weka/huongn/samples_dicts/filtered/mixed_no_repeat_min_comma_period_1_2"
TRAIN_STEPS=1000
EPOCH_STEPS=1000
EVAL_BATCH_SIZE=16
NUM_WORKERS=8
RUN_EVAL="False"
TRAIN_LOG_FREQ=2000
EVAL_FREQ=10000
CKPT_FREQ=10000

gantry run \
  --name ${EXP_NAME} \
  --task-name ${EXP_NAME} \
  --description "Filtering experiment transcripts that don’t have at least “,” or “.” or have repeating lines or are not in mixed-case or all" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --workspace ai2/open-whisper \
  --cluster ai2/neptune-cirrascale \
  --gpus ${GPU_COUNT} \
  --beaker-image huongn/ow_train_gantry \
  --pip requirements-main.txt \
  --budget ai2/prior \
  --priority normal \
  --weka oe-data-default:/weka \
  --dataset huongn/mini-job-ow-evalset:/ow_eval \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --env WANDB_DIR=/results/huongn/ow_logs \
  --env TORCH_NCCL_BLOCKING_WAIT=1 \
  --env NCCL_DEBUG=INFO \
  -- /bin/bash -c "torchrun --nnodes 1 --nproc_per_node ${GPU_COUNT} scripts/training/train.py \
      --model_variant=${MODEL_SIZE} \
      --exp_name=${EXP_NAME} \
      --job_type=${JOB_TYPE} \
      --samples_dicts_dir=${SAMPLES_DICTS_DIR} \
      --train_steps=${TRAIN_STEPS} \
      --epoch_steps=${EPOCH_STEPS} \
      --ckpt_file_name=None \
      --ckpt_dir=/weka/huongn/ow_ckpts \
      --log_dir=/results/huongn/ow_logs \
      --eval_dir=/ow_eval \
      --run_id_dir=/weka/huongn/ow_run_ids \
      --rank=None \
      --world_size=None \
      --lr=${LEARNING_RATE} \
      --betas='(0.9, 0.98)' \
      --eps=1e-6 \
      --weight_decay=0.1 \
      --max_grad_norm=1.0 \
      --eff_batch_size=${EFFECTIVE_BATCH_SIZE} \
      --train_batch_size=${BATCH_SIZE} \
      --val_batch_size=16 \
      --eval_batch_size=${EVAL_BATCH_SIZE} \
      --train_val_split=1.0 \
      --num_workers=8 \
      --pin_memory=True \
      --persistent_workers=True \
      --run_val=False \
      --run_eval=${RUN_EVAL} \
      --train_log_freq=${TRAIN_LOG_FREQ} \
      --val_freq=10000 \
      --eval_freq=${EVAL_FREQ} \
      --ckpt_freq=${CKPT_FREQ}"

# 2^20 = 1048576
# latest filter: 52740400 segments -> 206018 steps / epoch
