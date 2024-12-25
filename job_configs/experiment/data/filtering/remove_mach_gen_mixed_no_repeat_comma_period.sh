gantry run \
  --name "remove_mach_gen_mixed_no_repeat_comma_period" \
  --description "Remove transcripts that don’t have at least “,” or “.” or have repeating lines or are not in mixed-case or all (round 1 and 2 downloading (mod) only)" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --workspace ai2/open-whisper \
  --cluster ai2/neptune-cirrascale \
  --cpus 30 \
  --pip requirements-filter.txt \
  --budget ai2/prior \
  --replicas 20 \
  --priority normal \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/filtering/data_filter.py \
    --data_dir=/weka/huongn/ow_full \
    --samples_dicts_dir=/weka/huongn/samples_dicts/filtered/mixed_no_repeat_min_comma_period_4 \
    --batch_size=125 \
    --start_shard_idx=12449 \
    --end_shard_idx=14948 \
    --filter_mode=True \
    --metadata_path=/weka/huongn/samples_dicts/filtered/mixed_no_repeat_min_comma_period_4/metadata.txt \
    min_comma_period_mixed_no_repeat"

# 8449, 12448
# 2449, 8448
# 12449, 14948