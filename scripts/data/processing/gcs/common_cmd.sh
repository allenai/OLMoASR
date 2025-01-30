aws sqs create-queue --queue-name ow-seg --attributes MessageRetentionPeriod=1209600
aws sqs create-queue --queue-name ow-seg-ml --attributes MessageRetentionPeriod=1209600

#/Users/huongn/Desktop/open_whisper/logs/data/download/8M_en_text_only/shuffled_batches_1000.jsonl
aws sqs list-queues
aws sqs delete-queue --queue-url

python scripts/data/processing/gcs/create_instances.py \
    --iteration=None \
    --bucket=ow-seg \
    --project_id=oe-training \
    --service_account=349753783513-compute@developer.gserviceaccount.com \
    --vm_count=1 \
    --zone=us-west1-a \
    --machine_type=n2d-highcpu-48 \
    --termination_action=DELETE \
    --base_name=wo-woo \
    --queue_id=ow-seg \
    --tar_prefix="ow_4M_full" \
    --log_dir="seg_logs" \
    --seg_dir="segments" \
    --audio_dir="audio" \
    --transcript_only=False \
    --disk_size=50

# english
python scripts/data/processing/gcs/create_instances.py \
    --iteration=None \
    --bucket=ow-download-4m \
    --project_id=oe-training \
    --service_account=349753783513-compute@developer.gserviceaccount.com \
    --vm_count=1 \
    --zone=us-east1-b \
    --machine_type=n2d-highcpu-48 \
    --termination_action=DELETE \
    --base_name=ow-segment \
    --queue_id=ow-seg \
    --tar_prefix="ow_4M_full" \
    --log_dir="seg_logs" \
    --seg_dir="segments" \
    --audio_dir="audio" \
    --transcript_only=False \
    --disk_size=50

# multilingual - round 2
python scripts/data/processing/gcs/create_instances.py \
    --iteration=None \
    --bucket=ow-download-ml \
    --project_id=oe-training \
    --service_account=349753783513-compute@developer.gserviceaccount.com \
    --vm_count=50 \
    --zone=us-west1-a \
    --machine_type=n2d-highcpu-48 \
    --termination_action=DELETE \
    --base_name=ow-chunk-chunk \
    --queue_id=ow-seg-ml \
    --tar_prefix="ow_ml_full_rnd2" \
    --log_dir="seg_logs_rnd2" \
    --seg_dir="segments_rnd2" \
    --audio_dir="audio" \
    --transcript_only=False \
    --disk_size=50

# multilingual - round 1
python scripts/data/processing/gcs/create_instances.py \
    --iteration=None \
    --bucket=ow-download-ml \
    --project_id=oe-training \
    --service_account=349753783513-compute@developer.gserviceaccount.com \
    --vm_count=50 \
    --zone=asia-east1-a \
    --machine_type=n2d-highcpu-48 \
    --termination_action=DELETE \
    --base_name=ow-seg-seg-ml \
    --queue_id=ow-seg-ml-1 \
    --tar_prefix="ow_ml_full" \
    --log_dir="seg_logs" \
    --seg_dir="segments" \
    --audio_dir="audio" \
    --transcript_only=False \
    --disk_size=50

# multilingual - round 3
python scripts/data/processing/gcs/create_instances.py \
    --iteration=None \
    --bucket=ow-download-ml \
    --project_id=oe-training \
    --service_account=349753783513-compute@developer.gserviceaccount.com \
    --vm_count=50 \
    --zone=us-west1-a \
    --machine_type=n2d-highcpu-48 \
    --termination_action=DELETE \
    --base_name=ow-seg-ml-3 \
    --queue_id=ow-seg-ml-3 \
    --tar_prefix="ow_ml_full_rnd3" \
    --log_dir="seg_logs_rnd3" \
    --seg_dir="segments_rnd3" \
    --audio_dir="audio" \
    --transcript_only=False \
    --disk_size=50

python scripts/data/processing/gcs/create_instances.py \
    --iteration=None \
    --bucket=ow-download-ml \
    --project_id=oe-training \
    --service_account=349753783513-compute@developer.gserviceaccount.com \
    --vm_count=50 \
    --zone=northamerica-northeast1-a \
    --machine_type=n2d-highcpu-48 \
    --termination_action=DELETE \
    --base_name=ow-seg-seg-ml-3 \
    --queue_id=ow-seg-ml-3 \
    --tar_prefix="ow_ml_full_rnd3" \
    --log_dir="seg_logs_rnd3" \
    --seg_dir="segments_rnd3" \
    --audio_dir="audio" \
    --transcript_only=False \
    --disk_size=50

python scripts/data/processing/gcs/create_instances.py \
    --iteration=None \
    --bucket=ow-download-ml \
    --project_id=oe-training \
    --service_account=349753783513-compute@developer.gserviceaccount.com \
    --vm_count=50 \
    --zone=europe-north1-a \
    --machine_type=n2d-highcpu-48 \
    --termination_action=DELETE \
    --base_name=ow-chunk-ml-3 \
    --queue_id=ow-seg-ml-3 \
    --tar_prefix="ow_ml_full_rnd3" \
    --log_dir="seg_logs_rnd3" \
    --seg_dir="segments_rnd3" \
    --audio_dir="audio" \
    --transcript_only=False \
    --disk_size=50

python scripts/data/processing/gcs/create_instances.py \
    --iteration=None \
    --bucket=ow-download-ml \
    --project_id=oe-training \
    --service_account=349753783513-compute@developer.gserviceaccount.com \
    --vm_count=50 \
    --zone=asia-east1-a \
    --machine_type=n2d-highcpu-48 \
    --termination_action=DELETE \
    --base_name=ow-chunk-slay-ml-3 \
    --queue_id=ow-seg-ml-3 \
    --tar_prefix="ow_ml_full_rnd3" \
    --log_dir="seg_logs_rnd3" \
    --seg_dir="segments_rnd3" \
    --audio_dir="audio" \
    --transcript_only=False \
    --disk_size=50

python scripts/data/processing/gcs/create_instances.py \
    --iteration=None \
    --bucket=ow-seg \
    --project_id=oe-training \
    --service_account=349753783513-compute@developer.gserviceaccount.com \
    --vm_count=50 \
    --zone=asia-east1-a \
    --machine_type=n2d-highcpu-48 \
    --termination_action=DELETE \
    --base_name=ow-seg \
    --queue_id=ow-seg \
    --tar_prefix=None \
    --jsonl_prefix="jsonl_v2_simplefilter" \
    --manifest_prefix="local_seg_manifest" \
    --log_dir="seg_logs" \
    --seg_dir="segments" \
    --audio_dir=None \
    --transcript_only=True \
    --disk_size=50

# northamerica-northeast1-a - ow-chunk / ow-seg-ml
# us-south1-a - ow-seg-seg
# us-west1-a - ow-seg
# us-central1-a - ow-segmentation
# us-east1-b - ow-segment / ow-chunk-ml
# asia-east1-a - ow-seg-seg-ml
# europe-north1-a - ow-chunk-chunk

python scripts/data/processing/gcs/check_util.py --zone=us-south1-a --base_name=ow-seg-seg
python scripts/data/processing/gcs/check_util.py --zone=us-west1-a --base_name=ow-segmentation
python scripts/data/processing/gcs/check_util.py --zone=us-central1-a --base_name=ow-seg
python scripts/data/processing/gcs/check_util.py --zone=us-east1-b --base_name=ow-segment
python scripts/data/processing/gcs/check_util.py --zone=asia-east1-a --base_name=ow-chunking
python scripts/data/processing/gcs/check_util.py --zone=northamerica-northeast1-a --base_name=ow-chunk