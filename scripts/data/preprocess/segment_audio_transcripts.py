from open_whisper.preprocess import parallel_chunk_audio_transcript
import pandas as pd
import multiprocessing
from tqdm import tqdm
import os
from pathlib import Path
transcript_ext = "srt"
audio_ext = "m4a"

def main():
    with open("logs/data/download/transcript_paths.txt", "r") as f:
        transcript_file_paths = f.read().splitlines()

    with open("logs/data/download/audio_paths.txt", "r") as f:
        audio_file_paths = f.read().splitlines()

    transcript_file_paths = sorted(transcript_file_paths)
    audio_file_paths = sorted(audio_file_paths)

    with multiprocessing.Pool(multiprocessing.cpu_count() * 7) as pool:
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

if __name__ == "__main__":
    main()