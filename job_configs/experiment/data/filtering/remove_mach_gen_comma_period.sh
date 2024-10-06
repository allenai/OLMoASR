# gantry run \
#   --name "test_gantry" \
#   --allow-dirty \
#   --description "testing gantry" \
#   --workspace ai2/open-whisper \
#   --cluster ai2/neptune-cirrascale \
#   --beaker-image huongn/ow_filter_0 \
#   --cpus 5 \
#   --gpus 1 \
#   --pip requirements-filter.txt \
#   --env PYTHONPATH=/stage \
#   --no-nfs \
#   --preemptible \
#   --weka oe-data-default:/weka \
#   --budget ai2/prior \
#   --replicas 1 \
#   --venv base \
#   -- /bin/bash -c "python print('Hello, World!')"

gantry run --allow-dirty --no-nfs --preemptible --workspace ai2/open-whisper --cluster ai2/neptune-cirrascale --budget ai2/prior --weka oe-data-default:/weka -- python -c 'print("Hello, World!")'