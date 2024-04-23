#%%
import numpy as np
import os
# %%
with open("logs/data/download/sample_id_lang.txt", "r") as f:
    video_id_lang = {line.strip().split(" ")[0]:line.strip().split(" ")[1] for line in f}

with open("logs/data/download/sample_id_lang.txt", "r") as f:
    video_id_lst = [line.strip().split(" ")[0] for line in f] 

#%%
video_id_lst

#%%
video_ids = os.listdir("data/audio")
rng = np.random.default_rng(42)
sample_ids = rng.choice(video_ids, 100, replace=False).tolist()
# %%
sample_ids
# %%
import os
import shutil

def empty_directory(directory):
    # Check each file and sub-directory in the specified directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            # If it's a file or a symbolic link, delete it
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            # If it's a directory, delete the directory and all its contents
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
# %%
# moved the samples to data/sanity_check
for video_id in sample_ids:
    audio_dir = os.path.join("data/audio", video_id)
    transcript_dir = os.path.join("data/transcripts", video_id)
    empty_directory(audio_dir)
    shutil.move(audio_dir, "data/sanity_check/audio")
    empty_directory(transcript_dir)
    shutil.move(transcript_dir, "data/sanity_check/transcripts")
 
# %%
import multiprocessing
from open_whisper.preprocess import parallel_download_audio
from tqdm import tqdm
from itertools import repeat

with multiprocessing.Pool() as pool:
    out = list(
        tqdm(
            pool.imap_unordered(
                parallel_download_audio,
                zip(sample_ids, repeat("data/sanity_check/audio"), repeat("m4a")),
                chunksize=5,
            ),
            total=len(sample_ids),
        )
    )

#%%
sample_langs = [video_id_lang[video_id] for video_id in sample_ids]
sample_langs
# %%
from open_whisper.preprocess import parallel_download_transcript
with multiprocessing.Pool() as pool:
    out = list(
        tqdm(
            pool.imap_unordered(
                parallel_download_transcript,
                zip(
                    sample_ids,
                    sample_langs,
                    repeat("data/sanity_check/transcripts"),
                    repeat("srt"),
                ),
                chunksize=5,
            ),
            total=len(sample_ids),
        )
    )
# %%
os.listdir("data/sanity_check/transcripts")
# %%
