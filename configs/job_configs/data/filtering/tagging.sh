## tagging data ##
# document-level tagging
python scripts/data/filtering/data_tagger.py \
    --config_path configs/data_configs/tagging/main_tagging.yaml \
    --input_dir /path/to/jsonls_dir \
    --output_dir /path/to/output \
    --num_cpus 10
# segment-level tagging
python scripts/data/filtering/data_tagger.py \
    --config_path configs/data_configs/tagging/main_tagging_seg.yaml \
    --input_dir /path/to/jsonls_dir \
    --output_dir /path/to/output \
    --num_cpus 10
# audio language tagging
