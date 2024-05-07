#%%
import os
import shutil
from whisper.audio import load_audio
# %%
audio_files = []
for root, dirs, files in os.walk("/home/ubuntu/open_whisper/data/audio"):
    if "segments" in root:
        for f in os.listdir(root):
            audio_files.append(os.path.join(root, f))
audio_files
# %%
len(audio_files)
# %%
from tqdm import tqdm

# %%
import multiprocessing

#%%
def try_load_audio(f):
    try:
        load_audio(f)
    except:
        with open("logs/data/corrupted_audio.txt", "a") as fp:
            fp.write(f"{f}\n")
        return f
    return None

# %%
with multiprocessing.Pool() as pool:
    corrupted_audio = list(tqdm(pool.imap_unordered(try_load_audio, audio_files, chunksize=20), total=len(audio_files)))
# %%
with open("logs/data/preprocess/corrupted_audio.txt", "r") as f:
    corrupted_segments = [line.strip() for line in f]
# %%
corrupted_segments[:10]
# %%
corrupted_ids = list(set(["/".join(f.split("/")[:7]) for f in corrupted_segments]))
corrupted_ids
# %%
len(corrupted_ids)
# %%
