import multiprocessing
from tqdm import tqdm
from fire import Fire
import glob
import os
import gzip
import logging
import json
from itertools import repeat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def check_shard(shard_jsonl, manifest_file, dry_run):
    with gzip.open(shard_jsonl, "rt") as f:
        segments = [json.loads(line.strip()) for line in f]

    if "in_manifest" in segments[0]:
        if dry_run:
            not_valid_seg_count = sum(
                [1 for seg in segments if seg["in_manifest"] == False]
            )
            return not_valid_seg_count, len(segments)
        else:
            valid = [seg for seg in segments if seg["in_manifest"] == True]

            with gzip.open(shard_jsonl, "wt") as f:
                for seg in valid:
                    f.write(json.dumps(seg) + "\n")

            return len(valid), len(segments)
    else:
        with open(manifest_file, "r") as f:
            transcript_manifest = set([line.strip() for line in f])

        if dry_run:
            not_valid_seg_count = sum(
                [
                    1
                    for seg in segments
                    if "/".join(seg["subtitle_file"].split("/")[-2:])
                    not in transcript_manifest
                ]
            )
            return not_valid_seg_count, len(segments)
        else:
            valid = [
                seg
                for seg in segments
                if "/".join(seg["subtitle_file"].split("/")[-2:]) in transcript_manifest
            ]

            with gzip.open(shard_jsonl, "wt") as f:
                for seg in valid:
                    f.write(json.dumps(seg) + "\n")

            return len(valid), len(segments)


def parallel_check_shards(args):
    return check_shard(*args)


def main(source_dir, manifest_dir, dry_run):
    shard_jsonls = glob.glob(f"{source_dir}/*.jsonl.gz")
    get_shard = lambda shard_jsonl: shard_jsonl.split("_")[-1].split(".")[0]
    shards = [get_shard(shard_jsonl) for shard_jsonl in shard_jsonls]
    logger.info(f"{len(shards)} shards found")
    logger.info(f"{shards[:5]=}")
    manifest_files = [f"{manifest_dir}/{shard}.txt" for shard in shards]
    logger.info(f"{manifest_files[:5]=}")

    with multiprocessing.Pool() as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    parallel_check_shards,
                    zip(shard_jsonls, manifest_files, repeat(dry_run)),
                ),
                total=len(shard_jsonls),
            )
        )

    if dry_run:
        not_valid_seg_count, total_seg_count = zip(*results)
        not_valid_seg_count = sum(not_valid_seg_count)
        total_seg_count = sum(total_seg_count)
        logger.info(
            f"{not_valid_seg_count=}, {total_seg_count=}, {not_valid_seg_count/total_seg_count=}"
        )
    else:
        valid_seg_count, total_seg_count = zip(*results)
        valid_seg_count = sum(valid_seg_count)
        total_seg_count = sum(total_seg_count)
        logger.info(
            f"{valid_seg_count=}, {total_seg_count=}, {valid_seg_count/total_seg_count=}"
        )


if __name__ == "__main__":
    Fire(main)
