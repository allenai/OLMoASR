# gantry run \
#   --name "gen_sample_dicts" \
#   --description "Generate sample dict for training (from JSONL segmented data)" \
#   --allow-dirty \
#   --no-nfs \
#   --preemptible \
#   --workspace ai2/open-whisper \
#   --cluster ai2/neptune-cirrascale \
#   --cpus 62 \
#   --pip requirements-filter.txt \
#   --budget ai2/oe-data \
#   --replicas 20 \
#   --weka oe-data-default:/weka \
#   -- /bin/bash -c "python scripts/data/processing/gen_smpl_dict.py \
#     --shard_metadata="/weka/huongn/samples_dicts/unfiltered_rnd1_2M/unfiltered_rnd1_2M_shards.txt" \
#     --samples_dicts_dir=/weka/huongn/samples_dicts/unfiltered_rnd1_2M \
#     --batch_size=748 \
#     "

gantry run \
  --name "gen_sample_dicts" \
  --description "Generate sample dict for training (from JSONL segmented data)" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --workspace ai2/open-whisper \
  --cluster ai2/neptune-cirrascale \
  --cpus 62 \
  --pip requirements-filter.txt \
  --budget ai2/oe-data \
  --replicas 20 \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/processing/gen_smpl_dict.py \
    --jsonl_seg_dir="/weka/huongn/ow_seg_jsonl/v2_simplefilter" \
    --audio_dir="/weka/huongn/ow_seg" \
    --samples_dicts_dir="/weka/huongn/samples_dicts/filtered/v2_simplefilter" \
    --batch_size=748 \
    "