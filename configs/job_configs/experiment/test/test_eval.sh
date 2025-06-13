gantry run \
  --name "eval_debug" \
  --description "Debugging eval loop" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --workspace ai2/open-whisper \
  --cluster ai2/neptune-cirrascale \
  --cpus 23.25 \
  --gpus 1 \
  --beaker-image huongn/ow_train_gantry \
  --budget ai2/prior \
  --weka oe-data-default:/weka \
  --dataset huongn/mini-job-ow-evalset:/ow_eval \
  --priority normal \
  -- /bin/bash -c "tests/eval/test.py \
    --ckpt=/weka/huongn/tiny.en.pt \
    --eval_dir=/ow_eval
    "