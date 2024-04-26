#%%
import os
from open_whisper.utils import TranscriptReader
import multiprocessing
from tqdm import tqdm

#%%
all_transcripts = []
for root, *_ in os.walk("/mmfs1/gscratch/efml/hvn2002/open_whisper/data/transcripts"):
    if "segments" in root:
        all_transcripts.extend((os.path.join(root, path) for path in os.listdir(root)))
# %%
all_transcripts
# %%
len(all_transcripts)
# %%
def check_case(transcript_file):
    reader = TranscriptReader(transcript_file)
    t_dict, *_ = reader.read()
    text = reader.extract_text(t_dict)

    if text.islower():
        with open("/mmfs1/gscratch/efml/hvn2002/open_whisper/logs/filtering/machine_gen.txt", "a") as f:
            f.write(f"{transcript_file}\tLOWER\n")
        return transcript_file
    elif text.isupper():
        with open("/mmfs1/gscratch/efml/hvn2002/open_whisper/logs/filtering/machine_gen.txt", "a") as f:
            f.write(f"{transcript_file}\tUPPER\n")
        return transcript_file
    else:
        return None
# %%
check_case("/mmfs1/gscratch/efml/hvn2002/open_whisper/data/transcripts/_0gXV2Ew0Q0/segments/00:00:00.000_00:00:02.000.srt")
# %%
with multiprocessing.Pool() as pool:
    out = list(tqdm(pool.imap_unordered(check_case, all_transcripts, chunksize=20), total=len(all_transcripts)))
# %%
{path.split("/")[8] for path in out if path is not None}
# %%
len({path.split("/")[8] for path in out if path is not None})
# %%
len(os.listdir("/mmfs1/gscratch/efml/hvn2002/open_whisper/data/transcripts"))

# %%
