import multiprocessing
from collections import OrderedDict
from tqdm import tqdm
import gzip
import json
import os
import glob
from fire import Fire
from functools import partial
from itertools import repeat
from typing import Dict


def assign_audio_lang(output_dir: str, shard_file: str, shard_to_map_file: Dict):
    audio_lang_mapping_file = shard_to_map_file[shard_file]
    with gzip.open(audio_lang_mapping_file, "rt") as f:
        id_to_lang = json.load(f)
    with gzip.open(shard_file, "rt") as f:
        data = [json.loads(line.strip()) for line in f]

    for d in data:
        video_id = d["id"]
        if video_id in id_to_lang:
            d["audio_lang"] = id_to_lang[video_id]
        else:
            d["audio_lang"] = "en"
            
            with open(f"{output_dir}/missing_audio_lang.txt", "a") as f:
                f.write(f"{video_id}, {shard_file}, {audio_lang_mapping_file}\n")

    with gzip.open(f"{output_dir}/{os.path.basename(shard_file)}", "wt") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def parallel_assign_audio_lang(args, shard_to_map_file):
    return assign_audio_lang(*args, shard_to_map_file=shard_to_map_file)


def main(input_dir, output_dir, audio_lang_mapping_dir):
    range_list = [(i, i + 5) for i in range(0, 21196, 5)]
    flattened_range_list = [list(range(ele[0], ele[-1])) for ele in range_list]
    flattened_to_range_list = list(zip(flattened_range_list, range_list))
    shard_to_map_file = {
        f"{input_dir}/shard_{idx:08}.jsonl.gz": f"{audio_lang_mapping_dir}/ids_to_lang_{tpl[1][0]:08}_{tpl[1][-1]:08}.json.gz"
        for tpl in flattened_to_range_list
        for idx in tpl[0]
    }
    os.makedirs(output_dir, exist_ok=True)
    shard_files = glob.glob(f"{input_dir}/*.jsonl.gz")

    manager = multiprocessing.Manager()
    shared_dict = manager.dict(shard_to_map_file)
    parallel_assign_audio_lang_func = partial(
        parallel_assign_audio_lang, shard_to_map_file=shared_dict
    )

    with multiprocessing.Pool() as pool:
        _ = list(
            tqdm(
                pool.imap_unordered(
                    parallel_assign_audio_lang_func,
                    zip(
                        repeat(output_dir),
                        shard_files,
                    ),
                ),
                total=len(shard_files),
            ),
        )

if __name__ == "__main__":
    Fire(main)