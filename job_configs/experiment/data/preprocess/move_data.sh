gantry run \
  --name "move_data" \
  --description "Moving data on WEKA" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --workspace ai2/open-whisper \
  --cluster ai2/saturn-cirrascale \
  --cpus 69 \
  --replicas 20 \
  --priority normal \
  --pip requirements-filter.txt \
  --budget ai2/oe-data \
  --weka oe-data-default:/weka \
  -- /bin/bash -c "python scripts/data/processing/move_data.py \
    --paths_file="/weka/huongn/rnd4_full_move_back.txt" \
    --dry_run=False \
    --batch_size=51107
    "