import os
import glob

def get_data_shard(data_shard_idx: int):
    audio_files_train = sorted(glob.glob("data/" + f"{data_shard_idx:08d}/*/*.m4a"))
    transcript_files_train = sorted(glob.glob("data/" + f"{data_shard_idx:08d}/*/*.srt"))

    return audio_files_train, transcript_files_train