#%%
import os
# %%
transcript_files = []
for root, dirs, files in os.walk("/home/ubuntu/open_whisper/data/transcripts"):
    if "segments" in root:
        transcript_files.append(os.path.join(root, os.listdir(root)[0]))
transcript_files

# %%
import multiprocessing
from tqdm import tqdm

# %%
from open_whisper.utils import TranscriptReader

def check_machine_generated(file_path):
    reader = TranscriptReader(file_path)
    t_dict, *_ = reader.read()
    text = reader.extract_text(transcript=t_dict)
    fp = '/'.join(file_path.split("/")[:7])
    if text.isupper():
        with open("/home/ubuntu/open_whisper/logs/data/machine_generated.txt", "a") as f:
            f.write(f"{fp} UPPER\n")
        return fp
    elif text.islower():
        with open("/home/ubuntu/open_whisper/logs/data/machine_generated.txt", "a") as f:
            f.write(f"{fp} LOWER\n")
        return fp
    else:
        return None
# %%
with multiprocessing.Pool() as pool:
    results = list(tqdm(pool.imap_unordered(check_machine_generated, transcript_files, chunksize=20), total=len(transcript_files)))
# %%
check_machine_generated(file_path=transcript_files[0])
# %%
