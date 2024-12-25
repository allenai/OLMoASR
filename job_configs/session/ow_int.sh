beaker session create \
--gpus 6 \
--budget ai2/prior \
--bare \
--image beaker://huongn/ow_train_debug \
--cpus 62.0 \
--mount beaker://huongn/mini-job-ow-dataset=/wds_shards \
--mount beaker://huongn/mini-job-ow-evalset=/ow_eval \
--mount weka://oe-data-default=/weka \
--name ow_train_debug \
--priority normal \
--result /data/huongn/ow_logs \
--workspace ai2/open-whisper \
--env PYTHONPATH="/stage" \
--env WANDB_DIR=/data/huongn/ow_logs \
--secret-env WANDB_API_KEY=WANDB_API_KEY

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
--image beaker://huongn/data_processing \
--mount weka://oe-data-default=/weka \
--name ow_int_sesh \
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
--priority normal \
--result /data/huongn/ow_logs \
--workspace ai2/open-whisper \
--env PYTHONPATH="/stage" \
--env WANDB_DIR=/data/huongn/ow_logs \
--secret-env WANDB_API_KEY=WANDB_API_KEY