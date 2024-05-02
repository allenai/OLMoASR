from open_whisper.preprocess import (
    parallel_download_audio,
)
import pandas as pd
import multiprocessing
from tqdm import tqdm
from itertools import repeat
import os
import numpy as np
from datetime import datetime
from fire import Fire

def main(samples_file: str):
    """
    Download audio files from a file of samples.
    Has to be in the format of:
        `video_id \\t manual_caption_languages \\t automatic_caption_langage \\t duration`

    Parameters
    ----------
    samples_file: str
        Path to the file containing the samples

    Returns
    -------
    None
    """
    audio_ext = "m4a"
    with open(samples_file, "r") as f:
        sampled_en_list = [line.strip().split("\t")[0] for line in f]

    with multiprocessing.Pool() as pool:
        out = list(
            tqdm(
                pool.imap_unordered(
                    parallel_download_audio,
                    zip(sampled_en_list, repeat("data/audio"), repeat(audio_ext)),
                    chunksize=50,
                ),
                total=len(sampled_en_list),
            )
        )

if __name__ == "__main__":
    Fire(main)