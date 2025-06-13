beaker session create \
--gpus 6 \
--budget ai2/prior \
--bare \
--image beaker://huongn/ow_eval_debug \
--cpus 62.0 \
--mount beaker://huongn/mini-job-ow-dataset=/wds_shards \
--mount beaker://huongn/mini-job-ow-evalset=/ow_eval \
--mount weka://oe-data-default=/weka \
--name ow_eval_debug \
--priority normal \
--result /data/huongn/ow_logs \
--workspace ai2/open-whisper \
--env PYTHONPATH="/stage" \
--env WANDB_DIR=/data/huongn/ow_logs \
--secret-env WANDB_API_KEY=WANDB_API_KEY

beaker session create \
--budget ai2/oe-data \
--bare \
--image beaker://huongn/ow_data_viz \
--mount weka://oe-data-default=/weka \
--name huongn_int_sesh \
--priority normal \
--workspace ai2/open-whisper

beaker session create \
--budget ai2/prior \
--bare \
--image beaker://huongn/ow_train_debug \
--mount beaker://huongn/mini-job-ow-dataset=/wds_shards \
--mount beaker://huongn/mini-job-ow-evalset=/ow_eval \
--mount weka://oe-data-default=/weka \
--name ow_train_debug \
--priority low \
--result /data/huongn/ow_logs \
--workspace ai2/open-whisper \
--env PYTHONPATH="/stage" \
--env WANDB_DIR=/data/huongn/ow_logs \
--secret-env WANDB_API_KEY=WANDB_API_KEY \
--shared-memory 50GiB \
--gpus 1

beaker session create \
--gpus 1 \
--budget ai2/prior \
--bare \
--image beaker://huongn/ow_eval \
--mount beaker://huongn/mini-job-ow-dataset=/wds_shards \
--mount beaker://huongn/mini-job-ow-evalset=/ow_eval \
--mount weka://oe-data-default=/weka \
--name ow_eval_debug \
--priority normal \
--result /data/huongn/ow_logs \
--workspace ai2/open-whisper \
--env PYTHONPATH="/stage" \
--env WANDB_DIR=/data/huongn/ow_logs \
--secret-env WANDB_API_KEY=WANDB_API_KEY

beaker session create \
--gpus 1 \
--budget ai2/prior \
--bare \
--image beaker://huongn/ow_eval \
--mount weka://oe-data-default=/weka \
--name ow_eval_debug \
--priority high \
--workspace ai2/olmo3-webdata \
--env PYTHONPATH="/stage"

beaker session create \
--budget ai2/oe-data \
--bare \
--image beaker://huongn/ow_filter \
--mount weka://oe-data-default=/weka \
--name huongn_int_sesh_gpu \
--priority high \
--workspace ai2/open-whisper \
--gpus 1 \
--shared-memory 10G

beaker session create \
--budget ai2/oe-data \
--bare \
--image beaker://huongn/ow_data_processing \
--mount weka://oe-data-default=/weka \
--name huongn_debug_data_process \
--priority high \
--workspace ai2/open-whisper \
--shared-memory 20G \
--env PYTHONPATH="/stage"