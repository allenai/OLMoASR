import json
import gzip
import multiprocessing
from tqdm import tqdm
import glob
from typing import Optional, List, Literal
from fire import Fire
import os
from itertools import repeat


def process_scores(attributes_dict):
    scores = {}
    for tagger_info, score_info in attributes_dict.items():
        tagger = tagger_info.split("_quality")[0].split("ow_")[-1]
        score = score_info[0][-1]
        scores[tagger] = score

    return scores


def join_docs_and_attributes(docs_jsonl_gz, attributes_jsonl_gz, output_dir):
    assert (
        docs_jsonl_gz.split("_")[-1].split(".")[0]
        == attributes_jsonl_gz.split("_")[-1].split(".")[0]
    )
    docs = []
    with gzip.open(docs_jsonl_gz, "rt") as f:
        for line in f:
            docs.append(json.loads(line))

    attributes = []
    with gzip.open(attributes_jsonl_gz, "rt") as f:
        for line in f:
            attributes.append(json.loads(line))

    assert len(docs) == len(attributes)

    docs_attributes = zip(docs, attributes)

    shard_dict_list = []
    for doc, attribute in docs_attributes:
        shard_dict = {
            "subtitle_file": doc["metadata"]["subtitle_file"],
            "seg_content": doc["text"],
            "timestamp": doc["metadata"]["timestamp"],
            "id": doc["id"],
            "audio_file": doc["metadata"]["audio_file"],
            "fasttext_scores": process_scores(attribute["attributes"]),
        }
        shard_dict_list.append(shard_dict)

    with gzip.open(
        os.path.join(output_dir, os.path.basename(docs_jsonl_gz)), "wt"
    ) as f:
        for shard_dict in shard_dict_list:
            f.write(json.dumps(shard_dict) + "\n")


def parallel_join_docs_and_attributes(args):
    join_docs_and_attributes(*args)


def join_attributes(attributes_jsonl_gzs: List, output_dir: str):
    multiple_attributes_dict = []
    for attributes_jsonl_gz in attributes_jsonl_gzs:
        with gzip.open(attributes_jsonl_gz, "rt") as f:
            attributes_dict = [json.loads(line) for line in f]
        multiple_attributes_dict.append(attributes_dict)

    multiple_attributes_dict = list(zip(*multiple_attributes_dict))

    merged_attributes = []
    for attributes_dicts in multiple_attributes_dict:
        merged_attributes.append(
            {
                "id": attributes_dicts[0]["id"],
                "attributes": {
                    k: v
                    for attributes_dict in attributes_dicts
                    for k, v in attributes_dict["attributes"].items()
                },
            }
        )

    with gzip.open(
        os.path.join(output_dir, os.path.basename(attributes_jsonl_gzs[0])), "wt"
    ) as f:
        for attributes_dicts in merged_attributes:
            f.write(json.dumps(attributes_dicts) + "\n")


def parallel_join_attributes(args):
    join_attributes(*args)


def main(
    docs_dir: Optional[str],
    attributes_dirs: str,
    output_dir: str,
    mode: Literal["join_docs_and_attributes", "join_attributes"],
):
    os.makedirs(output_dir, exist_ok=True)
    if docs_dir is not None:
        shard_docs_jsonls = sorted(glob.glob(f"{docs_dir}/*.jsonl.gz"))
        print(f"{len(shard_docs_jsonls)} docs found")
        print(f"{shard_docs_jsonls[:5]=}")
        print(f"{shard_docs_jsonls[-5:]=}")

    if "," in attributes_dirs:
        shard_attributes_jsonls = []
        attributes_dirs = attributes_dirs.split(",")
        for attributes_dir in attributes_dirs:
            shard_attributes_jsonls.append(
                sorted(glob.glob(f"{attributes_dir}/*.jsonl.gz"))
            )
        shard_attributes_jsonls = list(zip(*shard_attributes_jsonls))
    else:
        shard_attributes_jsonls = sorted(glob.glob(f"{attributes_dirs}/*.jsonl.gz"))

    print(f"{len(shard_attributes_jsonls)} attributes found")
    print(f"{shard_attributes_jsonls[:5]=}")
    print(f"{shard_attributes_jsonls[-5:]=}")

    if mode == "join_attributes":
        with multiprocessing.Pool() as pool:
            _ = list(
                tqdm(
                    pool.imap_unordered(
                        parallel_join_attributes,
                        zip(shard_attributes_jsonls, repeat(output_dir)),
                    ),
                    total=len(shard_attributes_jsonls),
                )
            )
    else:
        with multiprocessing.Pool() as pool:
            _ = list(
                tqdm(
                    pool.imap_unordered(
                        parallel_join_docs_and_attributes,
                        zip(
                            shard_docs_jsonls,
                            shard_attributes_jsonls,
                            repeat(output_dir),
                        ),
                    ),
                    total=len(shard_docs_jsonls),
                )
            )


if __name__ == "__main__":
    Fire(main)
