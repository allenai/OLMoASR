# %%
import ray

# %%
ds = ray.data.read_text("data/00000", file_extensions=["srt"])
# %%
ds.schema()
# %%
ds.count()
# %%
ds = ray.data.read_text(
    "data/00000/_86o9kvgGZ4", file_extensions=["srt"], include_paths=True
)
# %%
ds.count()
# %%
ds.show(1)
# %%
import glob

# %%
glob.glob("data/00000/*/*.srt")
# %%
ds = ray.data.read_text(glob.glob("data/00000/*/*.srt"), file_extensions=["srt"])
# %%
ds.count()
# %%
ds.show(5)
# %%
ds = ray.data.read_binary_files(
    paths="data/00000", file_extensions=["srt"], include_paths=True
)
# %%
ds.show()
# %%
from typing import Dict, Any
import numpy as np


def bytes_to_text(transcript_dict: Dict[str, Any]) -> Dict[str, Any]:
    transcript_dict["text"] = transcript_dict["bytes"].decode("utf-8")
    del transcript_dict["bytes"]
    return transcript_dict


ds = ray.data.read_binary_files(
    paths="data/00000", file_extensions=["srt"], include_paths=True
).map(bytes_to_text)
# %%
ds.show()
# %%
ds.count()
# %%
ds = ray.data.read_binary_files(paths="data/00000/_86o9kvgGZ4", include_paths=True)

# %%
ds.schema()
# %%
ds.show(1)
# %%
ds.count()
# %%
# %%
ds = ray.data.read_binary_files(
    paths="data/00000/_86o9kvgGZ4", include_paths=True, override_num_blocks=1
)
# %%
ds.count()
# %%
ds.show()
# %%
ds.show()

# %%
import glob
from open_whisper.utils import TranscriptReader
import json
from typing import Optional, Literal
import numpy as np
import os


# %%
def check_case(transcript_dict: Dict[str, Any]) -> Dict[str, Any]:
    reader = TranscriptReader(transcript_string=transcript_dict["text"], ext="srt")
    t_dict, *_ = reader.read()
    text = reader.extract_text(t_dict)

    res_dict = {}
    res_dict["seg_dir"] = os.path.dirname(transcript_dict["path"]).replace(
        "440K_full", "440K_seg"
    )

    if text.islower():
        res_dict["label"] = "LOWER"
    elif text.isupper():
        res_dict["label"] = "UPPER"
    elif text == "":
        res_dict["label"] = "EMPTY"
    else:
        res_dict["label"] = "MIXED"

    return res_dict


# %%
ds = (
    ray.data.read_binary_files(
        paths="data/00000", file_extensions=["srt"], include_paths=True
    )
    .map(bytes_to_text)
    .map(check_case)
)

# %%
ds.show()
# %%
import multiprocessing
import multiprocess as mp
from tqdm import tqdm
from typing import List, Union


def filter_in_label(label_dict: Dict[str, Any], labels: Union[str, List[str]]) -> str:
    if isinstance(labels, str):
        if label_dict["label"] == labels:
            return label_dict["seg_dir"]
    elif isinstance(labels, list):
        if label_dict["label"] in labels:
            return label_dict["seg_dir"]
    else:
        return None


def parallel_filter_in_label(args) -> str:
    return filter_in_label(*args)


# %%
ds.repartition(num_blocks=1).write_json("logs/data/filtering/not_lower")
# %%
with open(glob.glob("logs/data/filtering/not_lower/*.json")[0], "r") as f:
    label_dicts = [json.loads(line) for line in f]
label_dicts
# %%
from itertools import repeat

with mp.Pool() as pool:
    res = list(
        tqdm(
            pool.imap_unordered(
                parallel_filter_in_label,
                zip(label_dicts, repeat(["MIXED", "EMPTY", "UPPER"])),
            ),
            total=len(label_dicts),
        )
    )

res
# %%
with open("logs/data/filtering/not_lower/not_lower.txt", "w") as f:
    for line in res:
        f.write(line + "\n")