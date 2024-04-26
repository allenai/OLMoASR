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
        with open("/mmfs1/gscratch/efml/hvn2002/open_whisper/logs/data/filtering/machine_gen.txt", "a") as f:
            f.write(f"{transcript_file}\tLOWER\n")
        return transcript_file
    elif text.isupper():
        with open("/mmfs1/gscratch/efml/hvn2002/open_whisper/logs/data/filtering/machine_gen.txt", "a") as f:
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
lower_t = []
upper_t = []
with open("/mmfs1/gscratch/efml/hvn2002/open_whisper/logs/data/filtering/machine_gen.txt", "r") as f:
    for line in f:
        path, case = line.strip().split("\t")
        if case == "LOWER":
            lower_t.append(path.split("/")[8])
        elif case == "UPPER":
            upper_t.append(path.split("/")[8])
    
# %%
print(len(lower_t))
lower_t = list(set(lower_t))
print(len(lower_t))
print(len(upper_t))
upper_t = list(set(upper_t))
print(len(upper_t))

#%%
lower_set = set(lower_t)
upper_set = set(upper_t)
all_machine_gen = list(lower_set.symmetric_difference(upper_set))
print(all_machine_gen)
# %%
import shutil
for video_id in all_machine_gen:
    t_path = os.path.join("/mmfs1/gscratch/efml/hvn2002/open_whisper/data/transcripts", video_id)
    t_dest = os.path.join("/mmfs1/gscratch/efml/hvn2002/open_whisper/data/machine_gen/transcripts")
    a_path = os.path.join("/mmfs1/gscratch/efml/hvn2002/open_whisper/data/audio", video_id)
    a_dest = os.path.join("/mmfs1/gscratch/efml/hvn2002/open_whisper/data/machine_gen/audio")

    shutil.move(t_path, t_dest)
    shutil.move(a_path, a_dest)
# %%
