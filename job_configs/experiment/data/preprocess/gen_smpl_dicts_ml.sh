WORKSPACE="ai2/vida" #ai2/open-whisper ai2/vida
GITHUB_TOKEN="HUONGN_GITHUB_TOKEN" #HUONGN_GITHUB_TOKEN GITHUB_TOKEN
PRIORITY="high"

gantry run \
  --name "gen_sample_dicts_ml" \
  --description "Generate sample dict (multilingual) for training" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --workspace ${WORKSPACE} \
  --cluster ai2/neptune-cirrascale \
  --replicas 5 \
  --cpus 15 \
  --pip requirements-filter.txt \
  --budget ai2/oe-data \
  --priority ${PRIORITY} \
  --weka oe-data-default:/weka \
  --gh-token-secret ${GITHUB_TOKEN} \
  -- /bin/bash -c "python scripts/data/processing/gen_smpl_dicts_ml.py \
    --src_dir="/weka/huongn/ow_seg_ml" \
    --split_factor=8 \
    --output_dir="/weka/huongn/samples_dicts/unfiltered_ml_1_2_3" \
    --batches=5
    "