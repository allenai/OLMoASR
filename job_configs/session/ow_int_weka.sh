beaker session create \
--budget ai2/oe-data \
--bare \
--image beaker://huongn/data_processing \
--mount weka://oe-data-default=/weka \
--name ow_int_sesh \
--priority normal \
--workspace ai2/open-whisper