## segmenting audio and transcript files into chunks ##
python olmoasr/preprocess.py/chunk_local \
    --transcript_file /path/to/transcript.vtt \
    --audio_file /path/to/audio.m4a \
    --output_dir /path/to/output \
    --in_memory False

## segmenting transcript files only ##
# transcript segments (not JSONL input)
python olmoasr/preprocess.py/chunk_local \
    --transcript_file /path/to/transcript.vtt \
    --audio_file /path/to/audio.m4a \
    --output_dir /path/to/output \
    --transcript_only True \
    --in_memory False

# transcript segments (manually-uploaded only)
python olmoasr/preprocess.py/preprocess_jsonl \
    --json_file /path/to/shard_00000.jsonl.gz \
    --shard 00000 \
    --log_dir /path/to/log \
    --output_dir /path/to/output \
    --in_memory True

# transcript segments (manually-uploaded + machine-transcribed)
python olmoasr/preprocess.py/preprocess_jsonl \
    --json_file /path/to/shard_00000.jsonl.gz \
    --shard 00000 \
    --log_dir /path/to/log \
    --output_dir /path/to/output \
    --seg_mach True \
    --in_memory True

# segmenting transcripts only + subsampling (manually-uploaded only)
python olmoasr/preprocess.py/preprocess_jsonl \
    --json_file /path/to/shard_00000.jsonl.gz \
    --shard 00000 \
    --log_dir /path/to/log \
    --output_dir /path/to/output \
    --subsample True \
    --subsample_size 20000 \
    --in_memory True

# segmenting transcripts only + subsampling (manually-uploaded + machine-transcribed)
python olmoasr/preprocess.py/preprocess_jsonl \
    --json_file /path/to/shard_00000.jsonl.gz \
    --shard 00000 \
    --log_dir /path/to/log \
    --output_dir /path/to/output \
    --subsample True \
    --subsample_size 20000 \
    --seg_mach True \
    --in_memory True

# randomly subsampling from segmented transcripts (JSONL containing transcript segments)
python olmoasr/preprocess.py/preprocess_jsonl \
    --json_file /path/to/shard_seg_00000.jsonl.gz \
    --shard 00000 \
    --log_dir /path/to/log \
    --output_dir /path/to/output \
    --only_subsample True \
    --subsample_size 20000

## segmenting audio files only ##
python olmoasr/preprocess.py/chunk_local \
    --transcript_file /path/to/transcript.vtt \
    --audio_file /path/to/audio.m4a \
    --output_dir /path/to/output \
    --audio_only True \
    --in_memory False