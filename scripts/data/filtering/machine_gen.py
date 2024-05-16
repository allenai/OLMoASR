#%%
import os
from open_whisper.utils import TranscriptReader
import multiprocessing
from tqdm import tqdm
import json
from typing import Optional, Literal, List
import numpy as np
import re
from itertools import repeat
#%%
id_to_segments = {}
for root, dirs, files in os.walk("data/transcripts"):
    if "segments" in root:
        id_to_segments[root.split("/")[2]] = [os.path.join(root, path) for path in os.listdir(root)]

# %%
def check_case(transcript_file: str, return_file_name: bool) -> Optional[Literal["LOWER", "UPPER"]]: 
    reader = TranscriptReader(transcript_file)
    t_dict, *_ = reader.read()
    text = reader.extract_text(t_dict)

    if text.islower():
        with open("logs/data/filtering/machine_gen.txt", "a") as f:
            f.write(f"{transcript_file}\tLOWER\n")
        return (transcript_file, "LOWER") if return_file_name else "LOWER"
    elif text.isupper():
        with open("logs/data/filtering/machine_gen.txt", "a") as f:
            f.write(f"{transcript_file}\tUPPER\n")
        return (transcript_file, "UPPER") if return_file_name else "UPPER"
    elif text == "":
        return (transcript_file, "EMPTY") if return_file_name else "EMPTY"
    else:
        return (transcript_file, "MIXED") if return_file_name else "MIXED"
    
# %%
id_to_casing = {}
for video_id, segments in tqdm(id_to_segments.items(), total=len(id_to_segments.keys())):
    with multiprocessing.Pool() as pool:
        result = list(tqdm(pool.imap_unordered(check_case, segments, chunksize=20), total=len(segments)))
    id_to_casing[video_id] = result

# %%
upper_count = 0
lower_count = 0
for video_id, casing in id_to_casing.items():
    if set(casing) == {"UPPER", "EMPTY"} or set(casing) == {"UPPER"}:
        upper_count += 1
        print("UPPER")
        print(video_id)
    elif set(casing) == {"LOWER", "EMPTY"} or set(casing) == {"LOWER"}:
        lower_count += 1
        print("LOWER")
        print(video_id)
print(f"{upper_count=}")
print(f"{lower_count=}")
# %%
prop_mixed_list = []
for video_id, casing in id_to_casing.items():
    mixed_count = sum([1 for c in casing if c is None])
    prop_mixed = np.round((mixed_count / len(casing)), 2) * 100
    prop_mixed_list.append((video_id, prop_mixed))
prop_mixed_list = sorted(prop_mixed_list, key=lambda x: x[1], reverse=True)

for video_id, prop_mixed in prop_mixed_list:
    print(f"{video_id}: {prop_mixed:.2f}%") 


# %%
only_lower = 0
only_lower_list = []
only_lower_set = set()
for video_id, casing in id_to_casing.items():
    if "LOWER" in casing and "UPPER" not in casing:
        only_lower += 1
        mixed_count = sum([1 for c in casing if c is None])
        prop_mixed = np.round((mixed_count / len(casing)), 2) * 100
        if prop_mixed == 0:
            only_lower_set.add(video_id)
        only_lower_list.append((video_id, prop_mixed))
print(f"{only_lower=}")
only_lower_list = sorted(only_lower_list, key=lambda x: x[1], reverse=False)
for video_id, prop_mixed in only_lower_list:
    print(f"{video_id}: {prop_mixed}%")
# %%
def check_punct(transcript_file: str, punct: str, return_file_name: bool = False) -> Optional[Literal["PUNCT", "NO PUNCT"]]:
    reader = TranscriptReader(transcript_file)
    t_dict, *_ = reader.read()
    text = reader.extract_text(t_dict)

    pattern = f"[{re.escape(punct)}]"

    if re.search(pattern, text):
        return (transcript_file, "PUNCT") if return_file_name else "PUNCT"
    elif text == "":
        return (transcript_file, "EMPTY") if return_file_name else "EMPTY"
    else:
        return (transcript_file, "NO PUNCT") if return_file_name else "NO PUNCT"

def parallel_check_punct(args) -> Optional[Literal["PUNCT", "NO PUNCT"]]:
    return check_punct(*args)
# %%
id_to_punct = {}
for video_id, segments in tqdm(id_to_segments.items(), total=len(id_to_segments.keys())):
    with multiprocessing.Pool() as pool:
        result = list(tqdm(pool.imap_unordered(parallel_check_punct, zip(segments, repeat("."), repeat(False)), chunksize=20), total=len(segments)))
    id_to_punct[video_id] = result
# %%
punct_count = 0
no_punct_count = 0
no_punct_set = set()
for video_id, punct_list in id_to_punct.items():
    if set(punct_list) == {"PUNCT", "EMPTY"} or set(punct_list) == {"PUNCT"}:
        punct_count += 1
        print("PUNCT")
        print(video_id)
    elif set(punct_list) == {"NO PUNCT", "EMPTY"} or set(punct_list) == {"NO PUNCT"}:
        no_punct_count += 1
        print("NO PUNCT")
        print(video_id)
        no_punct_set.add(video_id)
print(f"{punct_count=}")
print(f"{no_punct_count=}")
# %%
def detect_repeat(transcript_file: str, return_file_name: bool = False) -> Optional[Literal["REPEAT", "NO REPEAT"]]:
    reader = TranscriptReader(transcript_file)
    t_dict, *_ = reader.read()
    text = reader.extract_text(t_dict)

    all_text = [text for line in t_dict.values() for text in re.split(r'\n+', line.strip())]
    unique_text = set(all_text)

    if len(all_text) != len(unique_text):
        return (transcript_file, "REPEAT") if return_file_name else "REPEAT"
    elif text == "":
        return (transcript_file, "EMPTY") if return_file_name else "EMPTY"
    else:
        return (transcript_file, "NO REPEAT") if return_file_name else "NO REPEAT"

#%%
def parallel_detect_repeat(args) -> Optional[Literal["REPEAT", "NO REPEAT"]]:
    return detect_repeat(*args)
# %%
id_to_repeat = {}
for video_id, segments in tqdm(id_to_segments.items(), total=len(id_to_segments.keys())):
    with multiprocessing.Pool() as pool:
        result = list(tqdm(pool.imap_unordered(parallel_detect_repeat, zip(segments, repeat(False)), chunksize=20), total=len(segments)))
    id_to_repeat[video_id] = result
# %%
repeat_count = 0
no_repeat_count = 0
for video_id, repeat_list in id_to_repeat.items():
    if set(repeat_list) == {"REPEAT", "EMPTY"} or set(repeat_list) == {"REPEAT"}:
        repeat_count += 1
        print("REPEAT")
        print(video_id)
    elif set(repeat_list) == {"NO REPEAT", "EMPTY"} or set(repeat_list) == {"NO REPEAT"}:
        no_repeat_count += 1
        print("NO REPEAT")
        print(video_id)
print(f"{repeat_count=}")
print(f"{no_repeat_count=}")
# %%
