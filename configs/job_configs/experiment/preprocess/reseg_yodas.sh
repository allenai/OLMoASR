set -ex

for ((i=1; i<2; i++))
do
    SUBSET="en00$((0 + i))"
    echo "Processing subset: ${SUBSET}"
    gantry run \
        --name "reseg_yodas_${SUBSET}" \
        --description "resegmenting yodas" \
        --allow-dirty \
        --preemptible \
        --beaker-image huongn/ow_data_processing \
        --workspace ai2/open-whisper \
        --hostname saturn-cs-aus-$((230 + i)).reviz.ai2.in \
        --pip requirements/requirements-data.txt \
        --budget ai2/oe-data \
        --priority normal \
        --weka oe-data-default:/weka \
        -- /bin/bash -c "python scripts/data/processing/local/reseg_yodas.py \
            --input_dir=/weka/huongn/training_data/yodas/espnet___yodas/${SUBSET} \
            --audio_output_dir=/weka/huongn/training_data/reseg_yodas/${SUBSET}/audio \
            --text_output_dir=/weka/huongn/training_data/reseg_yodas/${SUBSET}/text
            "
    echo "Sleeping for 10 seconds"
    sleep 10
done