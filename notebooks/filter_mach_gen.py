# %%
import os
import glob
from open_whisper.utils import TranscriptReader
import multiprocess as mp
from tqdm import tqdm
import json
from typing import Optional, Literal, List
import numpy as np
import re
from itertools import repeat
import ray.data as rd


# %%
def check_case(
    transcript_file: str, return_file_name: bool
) -> Optional[Literal["LOWER", "UPPER", "MIXED", "EMPTY"]]:
    reader = TranscriptReader(transcript_file)
    t_dict, *_ = reader.read()
    text = reader.extract_text(t_dict)

    if text.islower():
        return (transcript_file, "LOWER") if return_file_name else "LOWER"
    elif text.isupper():
        return (transcript_file, "UPPER") if return_file_name else "UPPER"
    elif text == "":
        return (transcript_file, "EMPTY") if return_file_name else "EMPTY"
    else:
        return (transcript_file, "MIXED") if return_file_name else "MIXED"


def parallel_check_case(args) -> Optional[Literal["LOWER", "UPPER", "MIXED", "EMPTY"]]:
    return check_case(*args)


# %%
transcript_files = glob.glob("data/00000/*/*.srt")
with mp.Pool() as pool:
    res = list(
        tqdm(
            pool.imap_unordered(
                parallel_check_case,
                zip(transcript_files, repeat(True)),
            ),
            total=len(transcript_files),
        )
    )

# %%
lower_count = [1 for file, case in res if case == "LOWER"]
(sum(lower_count) / len(res)) * 100

# %%
lower_empty_count = [1 for file, case in res if case == "LOWER" or case == "EMPTY"]
(sum(lower_empty_count) / len(res)) * 100

# %%
with open(
    "/Users/huongn/Desktop/open_whisper/logs/data/filtering/transcript_casing.txt", "r"
) as f:
    transcript_casing = [line.strip().split(" ") for line in f]

transcript_casing


# %%
def not_lower(sample):
    if sample[1] != "LOWER":
        return sample
    else:
        return None


with mp.Pool() as pool:
    res = list(
        tqdm(
            pool.imap_unordered(not_lower, transcript_casing),
            total=len(transcript_casing),
        )
    )

not_lower = [sample for sample in res if sample is not None]

#%%
with open("/Users/huongn/Desktop/open_whisper/logs/data/filtering/not_lower.txt", "w") as f:
    for sample in not_lower:
        dir_path = "/".join(sample[0].split("/")[:6]).replace("ow_440K_tar", "440K_seg")
        f.write(f"{dir_path}\n")

# %%
def check_punct(
    transcript_file: str, punct: str, return_file_name: bool = False
) -> Optional[Literal["PUNCT", "NO PUNCT"]]:
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


# OW filtering idea added
# %%
def detect_repeat(
    transcript_file: str, return_file_name: bool = False
) -> Optional[Literal["REPEAT", "NO REPEAT"]]:
    reader = TranscriptReader(transcript_file)
    t_dict, *_ = reader.read()
    text = reader.extract_text(t_dict)

    all_text = [
        text for line in t_dict.values() for text in re.split(r"\n+", line.strip())
    ]
    unique_text = set(all_text)

    if len(all_text) != len(unique_text):
        return (transcript_file, "REPEAT") if return_file_name else "REPEAT"
    elif text == "":
        return (transcript_file, "EMPTY") if return_file_name else "EMPTY"
    else:
        return (transcript_file, "NO REPEAT") if return_file_name else "NO REPEAT"


# %%
def parallel_detect_repeat(args) -> Optional[Literal["REPEAT", "NO REPEAT"]]:
    return detect_repeat(*args)
