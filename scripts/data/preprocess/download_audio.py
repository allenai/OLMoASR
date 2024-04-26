from open_whisper.preprocess import (
    parallel_download_audio,
    standardize_dialects,
    detect_en,
)
import pandas as pd
import multiprocessing
from tqdm import tqdm
from itertools import repeat
import os
import numpy as np

audio_ext = "m4a"

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

metadata_files = []
for f in os.listdir("data/metadata"):
    metadata_files.append(os.path.join("data/metadata", f))

for f in metadata_files[1:]:
    print(f"Processing {f}")
    captions_df = pd.read_parquet(f)
    captions_df.reset_index(inplace=True)
    captions_df.drop("index", axis=1, inplace=True)
    captions_mod_df = captions_df.copy(deep=True)
    captions_mod_df["manual_caption_languages"] = captions_mod_df[
        "manual_caption_languages"
    ].apply(standardize_dialects)

    condition = (
        captions_mod_df["manual_caption_languages"]
        .str.split(",")
        .apply(lambda lst: set(lst) == {"en"})
    ) & (captions_mod_df["automatic_caption_orig_language"] == "en")
    temp_df = captions_mod_df[condition]
    rng = np.random.default_rng(42)
    sample_temp_idx = rng.choice(list(temp_df.index), size=6000, replace=False)

    with multiprocessing.Pool() as pool:
        idx_lst = list(
            tqdm(
                pool.imap_unordered(
                    detect_en, captions_mod_df.loc[sample_temp_idx].iterrows()
                ),
                total=len(captions_mod_df.loc[sample_temp_idx]),
            )
        )

    idx_lst = [idx for idx in idx_lst if idx is not None]
    sampled_en_idx = rng.choice(idx_lst, size=3800, replace=False)
    sampled_en_list = captions_df.loc[sampled_en_idx]["id"].to_list()

    with multiprocessing.Pool() as pool:
        out = list(
            tqdm(
                pool.imap_unordered(
                    parallel_download_audio,
                    zip(sampled_en_list, repeat("data/audio"), repeat(audio_ext)),
                    chunksize=5,
                ),
                total=len(sampled_en_list),
            )
        )

