import multiprocessing
from tqdm import tqdm
import tarfile
import os
from io import BytesIO
from typing import List, Tuple, Optional
import numpy as np
import glob
import json
import gzip
from itertools import repeat
import logging
from fire import Fire

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
)

logger = logging.getLogger(__name__)


def split_list(shard: str, segments: List, split_factor: int) -> List[Tuple]:
    rng = np.random.default_rng(42)
    rng.shuffle(segments)
    shard_idx = int(shard)
    start_shard_idx = shard_idx + ((split_factor - 1) * shard_idx)
    # Calculate size of each portion
    k, m = divmod(len(segments), split_factor)
    # Create each portion and append to the result
    segments_shards = [
        (
            segments[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)],
            f"{(start_shard_idx + i):08}",
        )
        for i in range(split_factor)
    ]

    uneven_shards = []
    if m > 0:
        uneven_shards = [
            segment_shard[-1]
            for segment_shard in segments_shards
            if len(segment_shard[0]) < k
        ]

    return segments_shards, uneven_shards


def process_jsonl(segments: List, shard: str, output_dir: str):
    tar_name = f"{shard}.tar.gz"
    tar_path = os.path.join(output_dir, tar_name)

    with tarfile.open(tar_path, "w:gz") as tar:
        for segment in segments:
            t_output_file = segment["subtitle_file"]
            a_output_file = segment["audio_file"].replace("ow_full", "ow_seg")
            transcript_string = segment["seg_content"]

            # Adding transcript to tar
            if t_output_file.endswith(".npy"):
                text_buffer = BytesIO()
                transcript_arr = np.array(transcript_string)
                np.save(text_buffer, transcript_arr)
                text_buffer.seek(0)
                tarinfo_text = tarfile.TarInfo(
                    name="/".join(t_output_file.split("/")[-2:])
                )
                tarinfo_text.size = text_buffer.getbuffer().nbytes
                tar.addfile(tarinfo_text, text_buffer)
            else:
                transcript_buffer = BytesIO()
                transcript_buffer.write(transcript_string.encode("utf-8"))
                transcript_buffer.seek(0)
                tarinfo_transcript = tarfile.TarInfo(
                    name="/".join(t_output_file.split("/")[-2:])
                )
                tarinfo_transcript.size = transcript_buffer.getbuffer().nbytes
                tar.addfile(tarinfo_transcript, transcript_buffer)

            # Adding audio array to tar
            tar.add(a_output_file, arcname="/".join(a_output_file.split("/")[-2:]))

    return tar_path


def write_to_tar(
    seg_jsonl: str, output_dir: str, subsampled_size: int, split_factor: int
) -> None:
    with gzip.open(seg_jsonl, "rt") as f:
        segments = [json.loads(line.strip()) for line in f]

    if len(segments) < subsampled_size:
        logger.info(
            f"{seg_jsonl} has less than {subsampled_size}: {len(segments)} segments"
        )

    shard = seg_jsonl.split("/")[-1].split(".")[0].split("_")[-1]

    if split_factor > 1:
        segments_shards, uneven_shards = split_list(shard, segments, split_factor)

        for segments, shard in segments_shards:
            process_jsonl(segments, shard, output_dir)

        if len(uneven_shards) > 0:
            [
                logger.info(f"Uneven shard: {shard} using split factor {split_factor}")
                for shard in uneven_shards
            ]

        return len(segments_shards), len(uneven_shards)
    else:
        process_jsonl(segments, shard, output_dir)

    return 1, 0


def parallel_write_to_tar(args):
    return write_to_tar(*args)


def main(
    input_dir: str,
    output_dir: str,
    subsampled_size: int,
    log_dir: str,
    split_factor: int,
    batch_size: Optional[int] = None,
):
    if batch_size:
        batch_idx = int(os.getenv("BEAKER_REPLICA_RANK"))
        seg_jsonls = sorted(glob.glob(f"{input_dir}/*.jsonl.gz"))[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    else:
        seg_jsonls = glob.glob(f"{input_dir}/*.jsonl.gz")
    logger.info(f"Processing {len(seg_jsonls)} jsonl files")
    logger.info(f"{seg_jsonls[:5]=}")
    logger.info(f"{seg_jsonls[-5:]=}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(f"{log_dir}/seg_jsonl_to_wds.log"))

    with multiprocessing.Pool() as pool:
        res = list(
            tqdm(
                pool.imap_unordered(
                    parallel_write_to_tar,
                    zip(
                        seg_jsonls,
                        repeat(output_dir),
                        repeat(subsampled_size),
                        repeat(split_factor),
                    ),
                ),
                total=len(seg_jsonls),
            )
        )

    full_shards, uneven_shards = zip(*res)
    logger.info(f"Full shards: {sum(full_shards)}")
    logger.info(f"Uneven shards: {sum(uneven_shards)}")


if __name__ == "__main__":
    Fire(main)
