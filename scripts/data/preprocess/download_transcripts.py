from open_whisper.preprocess import (
    parallel_download_transcript,
)
import pandas as pd
import multiprocessing
from tqdm import tqdm
from itertools import repeat
import os
import numpy as np
from datetime import datetime
from fire import Fire


def main(samples_file: str) -> None:
    """Download transcript files from a file of samples.

    Download transcript files from a file of samples.
    Has to be in the format of:
        `video_id \\t manual_caption_languages \\t automatic_caption_langage \\t duration`

    Args:
        samples_file: Path to the file containing the samples
    """
    transcript_ext = "srt"
    with open(samples_file, "r") as f:
        sample_id, sample_lang = list(
            zip(*[line.strip().split("\t")[:2] for line in f])
        )

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
                    chunksize=50,
                ),
                total=len(sample_id),
            )
        )


if __name__ == "__main__":
    Fire(main)
