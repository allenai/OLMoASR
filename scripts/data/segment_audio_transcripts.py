from open_whisper.preprocess import parallel_chunk_audio_transcript
import pandas as pd
import multiprocessing
from tqdm import tqdm
import os

transcript_ext = "srt"
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

transcript_file_paths = []
for root, dirs, files in os.walk("data/transcripts"):
    if len(root.split("/")) == 3:
        for f in files:
            transcript_file_paths.append(os.path.join(root, f))

audio_file_paths = []
for root, dirs, files in os.walk("data/audio"):
    if len(root.split("/")) == 3:
        for f in files:
            audio_file_paths.append(os.path.join(root, f))

transcript_file_paths = sorted(transcript_file_paths)
audio_file_paths = sorted(audio_file_paths)

with multiprocessing.Pool() as pool:
    out = list(
        tqdm(
            pool.imap_unordered(
                parallel_chunk_audio_transcript,
                zip(
                    transcript_file_paths,
                    audio_file_paths,
                ),
            ),
            total=len(transcript_file_paths),
        )
    )
