set -ex

FILTER="mixed_no_repeat_comma_period_mach_gen"
MODEL_SIZE="small"
LEARNING_RATE=1.5e-3
LR="15e4"
AUDIO_HOURS=440
BATCH_SIZE=16
EFFECTIVE_BATCH_SIZE=256
EPOCHS=5
DATE="121524"
EXP_NAME="${FILTER}_${MODEL_SIZE}_${LR}_${AUDIO_HOURS}K_bs${BATCH_SIZE}_ebs${EFFECTIVE_BATCH_SIZE}_${EPOCHS}pass_${DATE}"

GPU_COUNT=8
REPLICAS=2

JOB_TYPE="filtering"
SAMPLES_DICTS_DIR="/weka/huongn/samples_dicts/filtered/mixed_no_repeat_min_comma_period_full_1_2_3_4"
TRAIN_STEPS=1048576
EPOCH_STEPS=206699
EVAL_BATCH_SIZE=8
NUM_WORKERS=8
RUN_EVAL="True"
EVAL_SETS="librispeech_clean,librispeech_other"
TRAIN_LOG_FREQ=2000
EVAL_FREQ=10000
CKPT_FREQ=2000
VERBOSE="False"
DETECT_ANOMALY="False"
ADD_MODULE_HOOKS="True"
PRECISION="bfloat16"
USE_ORIG_PARAMS="False"

gantry run \
  --name ${EXP_NAME} \
  --task-name ${EXP_NAME} \
  --description "Filtering experiment transcripts that don’t have at least “,” or “.” or have repeating lines or are not in mixed-case or all" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --workspace ai2/open-whisper \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus ${GPU_COUNT} \
  --replicas ${REPLICAS} \
  --leader-selection \
  --propagate-failure \
  --synchronized-start-timeout "30m" \
  --host-networking \
  --beaker-image huongn/ow_train_gantry \
  --pip requirements-main.txt \
  --budget ai2/prior \
  --priority normal \
  --weka oe-data-default:/weka \
  --dataset huongn/mini-job-ow-evalset:/ow_eval \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --env-secret HF_TOKEN=HF_TOKEN \
  --env WANDB_DIR=/results/huongn/ow_logs \
  --env TORCH_NCCL_BLOCKING_WAIT=1 \
  --env NCCL_DEBUG=INFO \
  -- /bin/bash -c "torchrun --nnodes ${REPLICAS}:${REPLICAS} --nproc_per_node ${GPU_COUNT} --rdzv_id=101 --rdzv_backend=c10d --rdzv_endpoint=\${BEAKER_LEADER_REPLICA_HOSTNAME}:29400 scripts/training/train_fsdp.py \
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
      --eval_script_path=scripts/eval/eval.py \
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
      --eval_wandb_log=False \
      --eval_sets=${EVAL_SETS} \
      --train_log_freq=${TRAIN_LOG_FREQ} \
      --val_freq=10000 \
      --eval_freq=${EVAL_FREQ} \
      --ckpt_freq=${CKPT_FREQ} \
      --verbose=${VERBOSE} \
      --detect_anomaly=${DETECT_ANOMALY} \
      --add_module_hooks=${ADD_MODULE_HOOKS} \
      --precision=${PRECISION} \
      --use_orig_params=${USE_ORIG_PARAMS}"

# 2^20 = 1048576
# latest filter: 52740400 segments -> 206018 steps / epoch
