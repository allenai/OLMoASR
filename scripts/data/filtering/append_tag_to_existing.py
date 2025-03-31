import multiprocessing
from tqdm import tqdm
from itertools import repeat
import json
import gzip
import glob
import os
import yaml
from fire import Fire
from typing import List, Dict


def parse_config(config_path):
    config_dict = yaml.safe_load(open(config_path, "r"))
    return config_dict


def open_tagged_data_file(tagged_data_file) -> List[Dict]:
    with gzip.open(tagged_data_file, "rt") as f:
        data = [json.loads(line.strip()) for line in f]
    return data


def open_new_tag_data_file(new_tag_data_file) -> List[Dict]:
    with gzip.open(new_tag_data_file, "rt") as f:
        data = json.load(f)
    return data


def append_tags_to_sample(
    sample: Dict, new_tags_dict: Dict, add_key_mapping: Dict = None
) -> Dict:
    for tag, tag_value in new_tags_dict.items():
        if add_key_mapping and tag in add_key_mapping:
            sample[add_key_mapping[tag]] = tag_value
        else:
            sample[tag] = tag_value

    return sample


def append_tags_to_existing_data(
    tagged_data_file: str,
    new_tag_data_file: str,
    output_dir: str,
    add_key_mapping: Dict = None,
    seg_level: bool = False,
) -> None:
    # Load existing tagged data
    tagged_data = open_tagged_data_file(tagged_data_file)

    # Load new tag data
    new_tag_data = open_new_tag_data_file(new_tag_data_file)

    new_tagged_data = [
        append_tags_to_sample(
            sample,
            new_tag_data[sample["id" if seg_level == False else "seg_id"]],
            add_key_mapping,
        )
        for sample in tagged_data
    ]

    with gzip.open(
        os.path.join(output_dir, os.path.basename(tagged_data_file)), "wt"
    ) as f:
        for sample in new_tagged_data:
            f.write(json.dumps(sample) + "\n")


def parallel_append_tags(args):
    return append_tags_to_existing_data(*args)


def main(
    input_dir: str,
    new_tag_data_dir: str,
    output_dir: str,
    add_key_mapping_file: str = None,
    seg_level: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    tagged_data_files = sorted(glob.glob(input_dir + "/*.jsonl.gz"))
    new_tag_data_files = sorted(glob.glob(new_tag_data_dir + "/*.jsonl.gz"))

    print(f"{len(tagged_data_files)=}")
    print(f"{tagged_data_files[:5]=}")
    print(f"{tagged_data_files[-5:]=}")
    print(f"{len(new_tag_data_files)=}")
    print(f"{new_tag_data_files[:5]=}")
    print(f"{new_tag_data_files[-5:]=}")

    assert len(tagged_data_files) == len(new_tag_data_files), (
        f"Number of tagged data files {len(tagged_data_files)} "
        f"does not match number of new tag data files {len(new_tag_data_files)}"
    )

    if add_key_mapping_file:
        add_key_mapping = parse_config(add_key_mapping_file)
        print(f"{add_key_mapping=}")
    else:
        add_key_mapping = None
        print("No add_key_mapping_file provided, using default mapping.")

    with multiprocessing.Pool() as pool:
        _ = list(
            tqdm(
                pool.imap_unordered(
                    parallel_append_tags,
                    zip(
                        tagged_data_files,
                        new_tag_data_files,
                        repeat(output_dir),
                        repeat(add_key_mapping),
                        repeat(seg_level),
                    ),
                ),
                total=len(tagged_data_files),
            )
        )


if __name__ == "__main__":
    Fire(main)
