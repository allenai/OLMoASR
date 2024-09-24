beaker session create \
--budget ai2/oe-data \
--bare \
--image beaker://huongn/remove_lower_ray_test \
--mount weka://oe-data-default=/weka \
--name remove_lower_ray_session \
--priority normal \
--workspace ai2/open-whisper