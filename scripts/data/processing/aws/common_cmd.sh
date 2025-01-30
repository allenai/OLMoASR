python scripts/data/processing/aws/main.py --image_id=ami-09115b7bffbe3c5e4 \
    --name_pattern="ow-gen" \
    --instance_type=i4i.8xlarge \
    --key_name=huongn \
    --instance_count=30 \
    --bucket=allennlp-mattj \
    --bucket_prefix=openwhisper
sleep 120