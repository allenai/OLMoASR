gantry run \
  --name "move_data" \
  --description "Moving data on WEKA" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --workspace ai2/open-whisper \
  --cluster ai2/jupiter-cirrascale-2 \
  --cpus 69 \
  --gpus 3 \
  --replicas 20 \
  --priority normal \
  --pip requirements-filter.txt \
  --budget ai2/oe-data \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/processing/move_data.py \
    --paths_file="/weka/huongn/metadata/bad_mod_rnd_2_paths.txt" \
    --dry_run=False \
    --batch_size=21587
    "