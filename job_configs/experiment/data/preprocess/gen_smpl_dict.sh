gantry run \
  --name "gen_sample_dicts" \
  --description "Generate sample dict (unfiltered data) for training" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --workspace ai2/open-whisper \
  --cluster ai2/jupiter-cirrascale-2 \
  --cpus 62 \
  --gpus 2 \
  --pip requirements-filter.txt \
  --budget ai2/prior \
  --replicas 10 \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/processing/gen_smpl_dict.py \
    --shard_metadata="/weka/huongn/ow_filtering/sampled_shards.txt" \
    --samples_dicts_dir=/weka/huongn/ow_filtering/unfiltered \
    --batch_size=307 \
    "