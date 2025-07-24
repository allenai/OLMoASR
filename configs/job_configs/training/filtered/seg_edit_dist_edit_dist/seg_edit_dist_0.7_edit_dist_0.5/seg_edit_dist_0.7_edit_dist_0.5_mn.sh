set -ex

FILTER="seg_edit_dist_0.7_edit_dist_0.5_long" # <-- can be changed (threshold filter)
MODEL_SIZE="medium"
LEARNING_RATE=1.5e-3 # <-- might need to tune
LR="15e4" # <-- might need to tune
AUDIO_HOURS=440
BATCH_SIZE=16
GPU_COUNT=8
REPLICAS=4
ACCUMULATION_STEPS=1
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GPU_COUNT * REPLICAS * ACCUMULATION_STEPS))
EPOCHS=5
NUM_WORKERS=20 # <-- might need to tune
SHARDING_STRATEGY="FULL_SHARD"
EXP_NAME="${FILTER}_${MODEL_SIZE}_${LR}_${AUDIO_HOURS}K_bs${BATCH_SIZE}_ebs${EFFECTIVE_BATCH_SIZE}_${NUM_WORKERS}workers_${EPOCHS}pass"
echo ${EFFECTIVE_BATCH_SIZE}

SHARED_MEMORY="60.0GiB"
WORKSPACE="ai2/open-whisper"
CLUSTER="ai2/jupiter-cirrascale-2" # can be changed
HARDWARE="H100" # <-- can be changed
WANDB_API_KEY="WANDB_API_KEY"
HF_TOKEN="HF_TOKEN"
GITHUB_TOKEN="GITHUB_TOKEN"
PRIORITY="urgent"

JOB_TYPE="filtering"
SAMPLES_DICTS_DIR="/weka/huongn/training_data/filtered/filtered_from_tagged/seg_edit_dist_edit_dist/long/seg_edit_dist_0.7_edit_dist_0.5" # <-- can be changed (threshold filter)
EPOCH_STEPS=103068 # <-- change depending on dataset
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

TIMESTAMP_ON=True # <-- can be changed

# naming convention for exp_name

# timestamp switch
if [ "$TIMESTAMP_ON" == "True" ]; then
    EXP_NAME="${EXP_NAME}_TimestampOn"
    SCRIPT="scripts/training/prod/train_fsdp_timestamps.py"
else
    SCRIPT="scripts/training/prod/train_fsdp.py"
fi

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
DATE="042325"
EXP_NAME="${EXP_NAME}_${DATE}"

echo "Final EXP_NAME: $EXP_NAME"

gantry run \
  --name ${EXP_NAME} \
  --task-name ${EXP_NAME} \
  --description "${FILTER} training experiment" \
  --allow-dirty \
  --not-preemptible \
  --leader-selection \
  --host-networking \
  --propagate-failure \
  --propagate-preemption \
  --synchronized-start-timeout 15m \
  --workspace ${WORKSPACE} \
  --cluster ${CLUSTER} \
  --gpus ${GPU_COUNT} \
  --replicas ${REPLICAS} \
  --shared-memory ${SHARED_MEMORY} \
  --beaker-image huongn/ow_train_gantry \
  --pip requirements/requirements-train.txt \
  --budget ai2/prior \
  --priority ${PRIORITY} \
  --retries 3 \
  --weka oe-data-default:/weka \
  --dataset huongn/mini-job-ow-evalset:/ow_eval \
  --env-secret WANDB_API_KEY=${WANDB_API_KEY} \
  --env-secret HF_TOKEN=${HF_TOKEN} \
  --gh-token-secret ${GITHUB_TOKEN} \
  --env WANDB_DIR=/results/huongn/ow_logs \
  --env NCCL_DEBUG=INFO \
  --env NCCL_IB_HCA=^=mlx5_bond_0 \
  --env NCCL_SOCKET_IFNAME=ib \
  -- /bin/bash -c "torchrun --nnodes ${REPLICAS}:${REPLICAS} --nproc_per_node ${GPU_COUNT} --rdzv_id=101 --rdzv_backend=c10d --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400 ${SCRIPT} \
      --model_variant=${MODEL_SIZE} \
      --exp_name=${EXP_NAME} \
      --job_type=${JOB_TYPE} \
      --samples_dicts_dir=${SAMPLES_DICTS_DIR} \
      --train_steps=${TRAIN_STEPS} \
      --epoch_steps=${EPOCH_STEPS} \
      --ckpt_file_name=None \
      --ckpt_dir=/weka/huongn/ow_ckpts/filtered/tagged_data \
      --log_dir=/results/huongn/ow_logs \
      --eval_dir=/ow_eval \
      --run_id_dir=/weka/huongn/ow_run_ids/tagged_data \
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
      --sharding_strategy=${SHARDING_STRATEGY}"