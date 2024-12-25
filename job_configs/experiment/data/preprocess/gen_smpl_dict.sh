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
  --budget ai2/oe-data \
  --replicas 10 \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/processing/gen_smpl_dict.py \
    --shard_metadata="/weka/huongn/samples_dicts/unfiltered_rnd1_2M/unfiltered_rnd1_2M_shards.txt" \
    --samples_dicts_dir=/weka/huongn/samples_dicts/unfiltered_rnd1_2M \
    --batch_size=245 \
    "