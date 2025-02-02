set -ex

FILTER="text_heurs_1_manmach"
MODEL_SIZE="tiny"
LEARNING_RATE=1.5e-3
LR="15e4"
AUDIO_HOURS=440
BATCH_SIZE=128
GPU_COUNT=8
REPLICAS=1
ACCUMULATION_STEPS=1
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GPU_COUNT * REPLICAS * ACCUMULATION_STEPS))
EPOCHS=5
NUM_WORKERS=12
DATE="013025"
EXP_NAME="${FILTER}_${MODEL_SIZE}_${LR}_${AUDIO_HOURS}K_bs${BATCH_SIZE}_ebs${EFFECTIVE_BATCH_SIZE}_${NUM_WORKERS}workers_${EPOCHS}pass_${DATE}"
echo ${EFFECTIVE_BATCH_SIZE}

SHARED_MEMORY="40.0GiB"
WORKSPACE="ai2/open-whisper"
CLUSTER="ai2/ceres-cirrascale" # ai2/ceres-cirrascale
HARDWARE="H100"
WANDB_API_KEY="WANDB_API_KEY"
HF_TOKEN="HF_TOKEN"
GITHUB_TOKEN="GITHUB_TOKEN"
PRIORITY="normal"

JOB_TYPE="filtering"
SAMPLES_DICTS_DIR="/weka/huongn/samples_dicts/filtered/text_heurs_1_manmach_0.8_jan_25"
EPOCH_STEPS=102943
TRAIN_STEPS=524288
EVAL_BATCH_SIZE=8
RUN_EVAL="True"
TRAIN_LOG_FREQ=1000
EVAL_FREQ=5000
CKPT_FREQ=1000
VERBOSE="False"
PRECISION="float16"
SCRIPT="scripts/training/prod/train.py"

gantry run \
  --name ${EXP_NAME} \
  --task-name ${EXP_NAME} \
  --description "${FILTER} training experiment" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --workspace ${WORKSPACE} \
  --cluster ${CLUSTER} \
  --gpus ${GPU_COUNT} \
  --replicas ${REPLICAS} \
  --shared-memory ${SHARED_MEMORY} \
  --beaker-image huongn/ow_train_gantry \
  --pip requirements/requirements-train.txt \
  --budget ai2/prior \
  --priority ${PRIORITY} \
  --weka oe-data-default:/weka \
  --dataset huongn/mini-job-ow-evalset:/ow_eval \
  --env-secret WANDB_API_KEY=${WANDB_API_KEY} \
  --env-secret HF_TOKEN=${HF_TOKEN} \
  --gh-token-secret ${GITHUB_TOKEN} \
  --env WANDB_DIR=/results/huongn/ow_logs \
  --env NCCL_DEBUG=INFO \
  -- /bin/bash -c "torchrun --nnodes ${REPLICAS}:${REPLICAS} --nproc_per_node ${GPU_COUNT} ${SCRIPT} \
      --model_variant=${MODEL_SIZE} \
      --exp_name=${EXP_NAME} \
      --job_type=${JOB_TYPE} \
      --samples_dicts_dir=${SAMPLES_DICTS_DIR} \
      --train_steps=${TRAIN_STEPS} \
      --epoch_steps=${EPOCH_STEPS} \
      --ckpt_dir=/weka/huongn/ow_ckpts \
      --log_dir=/results/huongn/ow_logs \
      --eval_dir=/ow_eval \
      --run_id_dir=/weka/huongn/ow_run_ids \
      --lr=${LEARNING_RATE} \
      --betas='(0.9, 0.98)' \
      --eps=1e-6 \
      --weight_decay=0.1 \
      --max_grad_norm=1.0 \
      --eff_batch_size=${EFFECTIVE_BATCH_SIZE} \
      --train_batch_size=${BATCH_SIZE} \
      --eval_batch_size=${EVAL_BATCH_SIZE} \
      --num_workers=${NUM_WORKERS} \
      --pin_memory=True \
      --shuffle=True \
      --persistent_workers=True \
      --run_eval=${RUN_EVAL} \
      --train_log_freq=${TRAIN_LOG_FREQ} \
      --eval_freq=${EVAL_FREQ} \
      --ckpt_freq=${CKPT_FREQ} \
      --verbose=${VERBOSE} \
      --precision=${PRECISION} \
      --hardware=${HARDWARE}"
