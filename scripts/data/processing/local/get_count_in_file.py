import glob
import json
from typing import List, Dict
import multiprocessing
from tqdm import tqdm
from itertools import chain
import numpy as np
from fire import Fire
import gzip
from itertools import repeat


def get_count_file(dicts_file, get_dur) -> List[Dict]:
    with gzip.open(dicts_file, "rt") as f:
        data = [json.loads(line.strip()) for line in f]
        
    if get_dur:
        total_duration = sum([d["length"] for d in data])
        return len(data), total_duration
    else:
        return len(data)


def parallel_get_count_file(args):
    return get_count_file(*args)

def get_count(data_dir: str, get_dur: bool = False):
    dicts_files = glob.glob(f"{data_dir}/*.jsonl.gz")
    # samples_dicts_files = glob.glob(f"{samples_dicts_dir}/*/*.jsonl")
    # print(f"{samples_dicts_files=}\n")

    with multiprocessing.Pool() as pool:
        result = list(
            tqdm(
                pool.imap_unordered(parallel_get_count_file, zip(dicts_files, repeat(get_dur))),
                total=len(dicts_files),
            )
        )
    if get_dur:
        counts, durations = zip(*result)
        total_count = sum(counts)
        total_duration = sum(durations)
        return total_count, total_duration
    else:
        counts = result
        total_count = sum(counts)
        print(f"\n{total_count=}\n")
        return total_count


def get_duration(seg_count: int):
    duration = (seg_count * 30) / (60 * 60)
    return duration


def main(data_dir: str, get_dur: bool = False):
    if "seg" in data_dir:
        total_count = get_count(data_dir, get_dur)
        total_duration = get_duration(total_count)
        print(f"\n{total_count=}, {total_duration=}\n")
    else:
        get_dur = True
        total_count, total_duration = get_count(data_dir, get_dur)
        print(f"\n{total_count=}, {total_duration=}\n")


if __name__ == "__main__":
    Fire(main)
    # samples_dicts_dir = "/weka/huongn/samples_dicts/filtered/mixed_no_repeat_min_comma_period_full_1_2_3_4"
    # batch_size = 768
