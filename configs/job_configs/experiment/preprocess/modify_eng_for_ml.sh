WORKSPACE="ai2/vida" #ai2/open-whisper ai2/vida
GITHUB_TOKEN="HUONGN_GITHUB_TOKEN" #HUONGN_GITHUB_TOKEN GITHUB_TOKEN
PRIORITY="high"

gantry run \
  --name "modify_eng_for_ml" \
  --description "Modify English sample dict for multilingual training" \
  --allow-dirty \
  --no-nfs \
  --preemptible \
  --workspace ${WORKSPACE} \
  --cluster ai2/ceres-cirrascale \
  --replicas 10 \
  --cpus 15 \
  --pip requirements-filter.txt \
  --budget ai2/oe-data \
  --priority ${PRIORITY} \
  --weka oe-data-default:/weka \
  --gh-token-secret ${GITHUB_TOKEN} \
  -- /bin/bash -c "python scripts/data/processing/modify_eng_for_ml.py \
    --samples_dicts_dir="/weka/huongn/samples_dicts/filtered/mixed_no_repeat_min_comma_period_full_1_2_3_4" \
    --batches=10 \
    --output_dir="/weka/huongn/samples_dicts/unfiltered_ml_1_2_3" \
    --start_output_dir_idx=41
    "