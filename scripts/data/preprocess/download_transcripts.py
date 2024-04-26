from open_whisper.preprocess import (
    parallel_download_transcript,
    standardize_dialects,
    detect_en,
)
import pandas as pd
import multiprocessing
from tqdm import tqdm
from itertools import repeat
import os
import numpy as np
from tqdm import tqdm

transcript_ext = "srt"

# --- sanity-check example ---
# reading in metadata
# df = pd.read_parquet("data/metadata/captions-0010.parquet")
# only getting english data (that's less than 5 minutes long)
# en_df = df[
#     (df["manual_caption_languages"].str.contains("en"))
#     & (df["automatic_caption_orig_language"].str.contains("en"))
# ]

# rng = np.random.default_rng(42)

# sample = en_df[en_df["categories"] == "Education"][
#     ["id", "manual_caption_languages"]
# ].to_numpy()
# sample = rng.choice(sample, 10, replace=False)
# print(sample)

# # ensuring that language codes are english only
# for i, (id, langs) in enumerate(sample):
#     if "," in langs:
#         for lang in langs.split(","):
#             if "en" in lang:
#                 sample[i][1] = lang
#                 break

# metadata_files = []
# for f in os.listdir("data/metadata"):
#     metadata_files.append(os.path.join("data/metadata", f))

# df_list = []
# for f in metadata_files:
#     captions_df = pd.read_parquet(f)
#     df_list.append(
#         captions_df[
#             ["id", "manual_caption_languages", "automatic_caption_orig_language"]
#         ]
#     )

id_list = []
lang_list = []

for f in tqdm(os.listdir("logs/data")):
    if "en_id" in f:
        with open(f"logs/data/{f}", "r") as file:
            for line in file:
                id_list.append(line.strip().split(" ")[0])
                lang_list.append(line.strip().split(" ")[1])


sample_id = os.listdir("data/audio")
sample_lang = []


def get_lang_for_id(video_id):
    idx = id_list.index(video_id)
    langs = lang_list[idx]
    if "," in langs:
        lang = langs.split(",")[0]
    else:
        lang = langs
    with open("logs/data/sample_id_lang.txt", "a") as f:
        f.write(f"{video_id} {lang}\n")
    return (video_id, lang)


with multiprocessing.Pool() as pool:
    sample_id_lang = list(
        tqdm(
            pool.imap_unordered(get_lang_for_id, sample_id, chunksize=10),
            total=len(sample_id),
        )
    )

with open("logs/data/sample_id_lang.txt", "w") as f:
    for item in sample_id_lang:
        f.write(f"{item[0]} {item[1]}\n")

sample_id, sample_lang = zip(*sample_id_lang)
print(sample_id[:10])
print(sample_lang[:10])


with multiprocessing.Pool() as pool:
    out = list(
        tqdm(
            pool.imap_unordered(
                parallel_download_transcript,
                zip(
                    sample_id,
                    sample_lang,
                    repeat("data/transcripts"),
                    repeat(transcript_ext),
                ),
                chunksize=20,
            ),
            total=len(sample_id),
        )
    )
