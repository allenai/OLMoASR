import os
import multiprocessing
from tqdm import tqdm
import pandas as pd
import numpy as np
from open_whisper.preprocess import (
    standardize_dialects,
    detect_en,
)

metadata_files = []
for f in os.listdir("data/metadata"):
    metadata_files.append(os.path.join("data/metadata", f))

for f in tqdm(metadata_files):
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
    # english identified videos by metadata
    eng_by_md_df = captions_mod_df[condition]

    captions_df.loc[eng_by_md_df.index][
        ["id", "manual_caption_languages", "automatic_caption_orig_language", "duration"]
    ].to_csv(
        f"data/english_only/{f.split('/')[-1].split('.')[0]}.txt",
        sep="\t",
        index=False,
        header=False,
    )

    

