gantry run \
  --name "remove_mach_gen_mixed_no_repeat_comma_period" \
  --description "Remove transcripts that don’t have at least “.” or “.” or have repeating lines or are not in mixed-case or all" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --workspace ai2/open-whisper \
  --cluster ai2/neptune-cirrascale \
  --cpus 62 \
  --gpus 2 \
  --pip requirements-filter.txt \
  --budget ai2/prior \
  --replicas 10 \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/filtering/data_filter.py \
    --data_dir=/weka/huongn/440K_full \
    --samples_dicts_dir=/weka/huongn/ow_filtering/mixed_no_repeat_min_comma_period \
    --batch_size=245 \
    --filter_mode=True \
    --metadata_path=/weka/huongn/ow_filtering/mixed_no_repeat_min_comma_period/metadata.txt \
    min_comma_period_mixed_no_repeat"
  
# gantry run \
#   --name "test_gantry" \
#   --description "testing gantry" \
#   --allow-dirty \
#   --no-nfs \
#   --preemptible \
#   --beaker-image huongn/ow_filter_3 \
#   --workspace ai2/open-whisper \
#   --cluster ai2/neptune-cirrascale \
#   --cpus 5 \
#   --gpus 1 \
#   --venv base \
#   --pip requirements-filter.txt \
#   --budget ai2/prior \
#   --replicas 10 \
#   --weka oe-data-default:/weka \
#   -- /bin/bash -c "python /stage/scripts/data/filtering/data_filter.py \
#     --data_dir=/weka/huongn/440K_full \
#     --samples_dicts_dir=/weka/huongn/ow_filtering/test_gantry_2/$(printf '%03d' $((BEAKER_REPLICA_RANK))) \
#     --batch_size=245 \
#     --batch_idx=$((BEAKER_REPLICA_RANK)) \
#     --filter_mode=True \
#     --metadata_path=/weka/huongn/ow_filtering/test_gantry_2/metadata.txt \
#     no_lower_no_repeat"