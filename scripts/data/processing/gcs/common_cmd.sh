aws sqs create-queue --queue-name ow-seg --attributes MessageRetentionPeriod=1209600

#/Users/huongn/Desktop/open_whisper/logs/data/download/8M_en_text_only/shuffled_batches_1000.jsonl
aws sqs list-queues
aws sqs delete-queue --queue-url

python scripts/data/processing/gcs/create_instances.py \
    --iteration=None \
    --bucket=ow-download-4m \
    --project_id=oe-training \
    --service_account=349753783513-compute@developer.gserviceaccount.com \
    --vm_count=1 \
    --zone=us-central1-a \
    --machine_type=n2d-highcpu-48 \
    --termination_action=DELETE \
    --base_name=ow-seg-seg \
    --queue_id=ow-seg \
    --tar_prefix="ow_4M_full" \
    --log_dir="seg_logs" \
    --seg_dir="segments" \
    --audio_dir="audio" \
    --disk_size=50

# northamerica-northeast1-a - ow-chunk
# us-south1-a - ow-seg-seg
# us-west1-a - ow-segmentation
# us-central1-a - ow-seg
# us-east1-b - ow-segment
# asia-east1-a - ow-chunking
# europe-north1-a - ow-chunk-chunk

python scripts/data/processing/gcs/check_util.py --zone=us-south1-a --base_name=ow-seg-seg
python scripts/data/processing/gcs/check_util.py --zone=us-west1-a --base_name=ow-segmentation
python scripts/data/processing/gcs/check_util.py --zone=us-central1-a --base_name=ow-seg
python scripts/data/processing/gcs/check_util.py --zone=us-east1-b --base_name=ow-segment
python scripts/data/processing/gcs/check_util.py --zone=asia-east1-a --base_name=ow-chunking
python scripts/data/processing/gcs/check_util.py --zone=northamerica-northeast1-a --base_name=ow-chunk