from open_whisper.preprocess import parallel_chunk_audio_transcript
import pandas as pd
import multiprocessing
from tqdm import tqdm

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

print("Reading in data")
captions_0000 = pd.read_parquet("data/metadata/captions-0000.parquet")
en_df = captions_0000[
    (captions_0000["manual_caption_languages"].str.contains("en"))
    & (captions_0000["automatic_caption_orig_language"].str.contains("en"))
]

hq_df = en_df[
    (en_df["categories"] == "Science & Technology")
    | (en_df["categories"] == "Education")
    | (en_df["categories"] == "News & Politics")
].sort_values(by="view_count", ascending=False)[:25000][
    ["id", "manual_caption_languages"]
]

sample = hq_df.to_numpy()

# ensuring that language codes are english only
for i, (id, langs) in enumerate(sample):
    if "," in langs:
        for lang in langs.split(","):
            if "en" in lang:
                sample[i][1] = lang
                break


sample_id, sample_lang = [row[0] for row in sample], [row[1] for row in sample]

# transcript and audio file paths for reference when chunking
transcript_file_paths = [
    f"data/transcripts/{sample_id[i]}/{sample_id[i]}.{sample_lang[i]}.{transcript_ext}"
    for i in range(len(sample_id))
]

audio_file_paths = [f"data/audio/{id}/{id}.{audio_ext}" for id in sample_id]

# debugging chunking
# for i in range(0, len(sample)):
#     print(sample[i][0])
#     chunk_audio_transcript(
#         transcript_file_paths[i], audio_file_paths[i], transcript_ext
#     )

# debugging chunking
# chunk_audio_transcript(transcript_file_paths[7], audio_file_paths[7], transcript_ext)

# debugging chunking - for segment checking
# with multiprocessing.Pool() as pool:
#     out = list(
#         tqdm(
#             pool.imap_unordered(
#                 parallel_chunk_audio_transcript,
#                 zip(
#                     sample_transcript_file_paths,
#                     sample_audio_file_paths,
#                     repeat(transcript_ext),
#                 ),
#             ),
#             total=30,
#         )
#     )

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
            total=len(sample),
        )
    )
