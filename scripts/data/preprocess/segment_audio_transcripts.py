from open_whisper.preprocess import parallel_chunk_audio_transcript
import pandas as pd
import multiprocessing
from tqdm import tqdm
import os

transcript_ext = "srt"
audio_ext = "m4a"

def main():
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
                    chunksize=20,
                ),
                total=len(transcript_file_paths),
            )
        )

if __name__ == "__main__":
    main()