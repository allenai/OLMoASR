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
from typing import Optional


def get_bigsubset(line: str, current_ids, failed_ids):
    row = line.strip().split("\t")
    if row[0] not in current_ids and row[0] not in failed_ids:
        lang = row[1].split(",")[0]
        row[1] = lang
        return row


def parallel_get_bigsubset(args):
    return get_bigsubset(*args)


def main(captions_idx: Optional[int]):
    audio_ext = "m4a"
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%m-%d")
    # randomly choosing the file to sample from
    file_list = sorted(
        [os.path.join("data/english_only", f) for f in os.listdir("data/english_only")]
    )

    if captions_idx is None:
        rng = np.random.default_rng(42)
        file_path = rng.choice(file_list, 1, replace=False)[0]
    else:
        file_path = file_list[captions_idx]
        print(f"{file_path=}")

    if len(os.listdir("data/audio")) != 0:
        # getting all current videos downloaded and ones where audio or text can't be extracted so no double downloading
        current_ids = set(os.listdir("data/audio") + os.listdir("data/transcripts"))
    else:
        current_ids = []

    if os.path.exists("logs/data/download/failed_download_a.txt"):
        with open("logs/data/download/failed_download_a.txt", "r") as f:
            failed_ids_a = [line.strip() for line in f]
    else:
        failed_ids_a = []

    if os.path.exists("logs/data/download/failed_download_t.txt"):
        with open("logs/data/download/failed_download_t.txt", "r") as f:
            failed_ids_t = [line.strip() for line in f]
    else:
        failed_ids_t = []

    failed_ids = list(set(failed_ids_a + failed_ids_t))

    # getting data from file to sample from that isn't in current_ids or failed_ids and getting only 1st manual caption language code
    with open(file_path, "r") as f:
        all_lines = f.readlines()

    with multiprocessing.Pool() as pool:
        big_subset = list(
            tqdm(
                pool.imap_unordered(
                    parallel_get_bigsubset,
                    zip(all_lines, repeat(current_ids), repeat(failed_ids)),
                    chunksize=50,
                ),
                total=len(all_lines),
            )
        )
    big_subset = [item for item in big_subset if item is not None]

    rng = np.random.default_rng(42)
    sampled_en_list = list(rng.choice(big_subset, 100000, replace=False))

    # recording the sampled data to get numbers of initial download
    # with open(f"logs/data/download/sampled_en_{formatted_datetime}_{file_path.split("/")[-1].split(".")[0]}.txt", "w") as f:
    with open(f"logs/data/download/sampled_en_{formatted_datetime}.txt", "a") as f:
        for item in sampled_en_list:
            f.write("\t".join(item))
            f.write("\n")


if __name__ == "__main__":
    Fire(main)
