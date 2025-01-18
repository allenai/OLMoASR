import os
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
import time
import multiprocessing
from fire import Fire
from itertools import repeat, chain
import numpy as np
import sys
import glob
import json
import gzip

sys.path.append(os.getcwd())
import scripts.data.processing.segment_jsonl_utils as utils
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.FileHandler("preprocess_gcs.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

SEGMENT_COUNT_THRESHOLD = 120


def unarchive_jsonl_gz(file_path, output_path=None):
    """
    Unarchive a .jsonl.gz file and optionally save the uncompressed content to an output file.

    Args:
        file_path (str): Path to the .jsonl.gz file.
        output_path (str, optional): Path to save the uncompressed .jsonl file. Defaults to None.

    Returns:
        list[dict]: A list of JSON objects parsed from the .jsonl file.
    """
    data = []
    with gzip.open(file_path, "rt", encoding="utf-8") as gz_file:
        data = [json.loads(line.strip()) for line in gz_file]

    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_file:
            for json_obj in data:
                out_file.write(json.dumps(json_obj) + "\n")

    return data


def chunk_transcript(
    transcript_data: Dict,
    transcript_manifest: List[str],
    log_dir: str,
    in_memory: bool = True,
) -> Optional[List[Tuple[str, str, str, np.ndarray]]]:
    """Segment audio and transcript files into <= 30-second chunks

    Segment audio and transcript files into <= 30-second chunks. The audio and transcript files are represented by audio_file and transcript_file respectively.

    Args:
    transcript_file: Path to the transcript file
    audio_file: Path to the audio file

    Raises:
        Exception: If an error occurs during the chunking process
    """
    try:
        transcript_string = transcript_data["content"]
        transcript_file = transcript_data["subtitle_file"]
        print(f"{transcript_file=}")
        if transcript_file.startswith("/weka"):
            video_id = transcript_file.split("/")[5]
        else:
            video_id = transcript_file.split("/")[1]
        print(f"{video_id=}")

        output_dir = os.path.dirname(transcript_file)
        transcript_ext = transcript_file.split(".")[-1]
        segment_count = 0

        transcript, *_ = utils.TranscriptReader(
            file_path=None, transcript_string=transcript_string, ext=transcript_ext
        ).read()

        if len(transcript.keys()) == 0:
            with open(f"{log_dir}/empty_transcripts.txt", "a") as f:
                f.write(f"{video_id}\n")
            return None

        a = 0
        b = 0

        timestamps = list(transcript.keys())
        diff = 0
        init_diff = 0
        segments_list = []

        while a < len(transcript) + 1 and segment_count < SEGMENT_COUNT_THRESHOLD:
            init_diff = utils.calculate_difference(timestamps[a][0], timestamps[b][1])

            if init_diff < 30000:
                diff = init_diff
                b += 1
            else:
                # edge case (when transcript line is > 30s)
                if b == a:
                    with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                        f.write(f"{video_id}\tindex: {b}\n")

                    a += 1
                    b += 1

                    if a == b == len(transcript):
                        if segment_count == 0:
                            with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                                f.write(f"{video_id}\tdelete\n")
                        break

                    continue

                over_ctx_len, err = utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript, language=None
                )
                if not over_ctx_len:
                    t_output_file, transcript_string = utils.write_segment(
                        timestamps=timestamps[a:b],
                        transcript=transcript,
                        output_dir=output_dir,
                        ext=transcript_ext,
                        in_memory=in_memory,
                    )

                    if not utils.too_short_audio_text(
                        start=timestamps[a][0], end=timestamps[b - 1][1]
                    ):
                        timestamp = t_output_file.split("/")[-1].split(
                            f".{transcript_ext}"
                        )[0]
                        segments_list.append(
                            {
                                "subtitle_file": t_output_file,
                                "seg_content": transcript_string,
                                "timestamp": timestamp,
                                "id": video_id,
                                "audio_file": t_output_file.replace(
                                    f".{transcript_ext}", ".npy"
                                ),
                            }
                        )
                        segment_count += 1
                else:
                    if err is not None:
                        with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")
                    else:
                        with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")

                init_diff = 0
                diff = 0

                # checking for silence
                if timestamps[b][0] > timestamps[b - 1][1]:
                    silence_segments = (
                        utils.calculate_difference(
                            timestamps[b - 1][1], timestamps[b][0]
                        )
                        // 30000
                    )

                    for i in range(0, silence_segments + 1):
                        start = utils.adjust_timestamp(
                            timestamps[b - 1][1], (i * 30000)
                        )

                        if i == silence_segments:
                            if start == timestamps[b][0]:
                                continue
                            else:
                                end = timestamps[b][0]
                        else:
                            end = utils.adjust_timestamp(start, 30000)

                        t_output_file, transcript_string = utils.write_segment(
                            timestamps=[(start, end)],
                            transcript=None,
                            output_dir=output_dir,
                            ext=transcript_ext,
                            in_memory=in_memory,
                        )

                    if not utils.too_short_audio_text(start=start, end=end):
                        segments_list.append(
                            {
                                "subtitle_file": t_output_file,
                                "seg_content": transcript_string,
                                "timestamp": timestamp,
                                "id": video_id,
                                "audio_file": t_output_file.replace(
                                    f".{transcript_ext}", ".npy"
                                ),
                            }
                        )
                        segment_count += 1
                a = b

            if b == len(transcript) and diff < 30000:
                over_ctx_len, err = utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript, language=None
                )
                if not over_ctx_len:
                    t_output_file, transcript_string = utils.write_segment(
                        timestamps=timestamps[a:b],
                        transcript=transcript,
                        output_dir=output_dir,
                        ext=transcript_ext,
                        in_memory=in_memory,
                    )

                    if not utils.too_short_audio_text(
                        start=timestamps[a][0], end=timestamps[b - 1][1]
                    ):
                        segments_list.append(
                            {
                                "subtitle_file": t_output_file,
                                "seg_content": transcript_string,
                                "timestamp": timestamp,
                                "id": video_id,
                                "audio_file": t_output_file.replace(
                                    f".{transcript_ext}", ".npy"
                                ),
                            }
                        )
                        segment_count += 1
                else:
                    if err is not None:
                        with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")
                    else:
                        with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")

                break
        if len(segments_list) == 0:
            return None

        segments_list = [
            segment
            for segment in segments_list
            if "/".join(segment["subtitle_file"].split("/")[-2:]) in transcript_manifest
        ]
        return segments_list
    except ValueError as e:
        with open(f"{log_dir}/failed_chunking.txt", "a") as f:
            f.write(f"{video_id}\t{e}\n")
        return None
    except Exception as e:
        with open(f"{log_dir}/failed_chunking.txt", "a") as f:
            f.write(f"{video_id}\t{e}\n")
        return None


def parallel_chunk_transcript(
    args,
) -> Optional[List[Tuple[str, str, str, np.ndarray]]]:
    """Parallelized version of chunk_audio_transcript function to work in multiprocessing context"""
    return chunk_transcript(*args)


def preprocess_jsonl(
    json_file: str,
    shard: str,
    transcript_manifest_file: str,
    log_dir: str,
    output_dir: str,
    subsample: bool = False,
    subsample_size: int = 0,
    subsample_seed: int = 42,
    in_memory: bool = True,
):
    log_dir = os.path.join(log_dir, shard)
    os.makedirs(log_dir, exist_ok=True)

    if json_file.endswith(".gz"):
        transcript_data = unarchive_jsonl_gz(json_file)
    else:
        with open(json_file, "r") as f:
            transcript_data = [json.loads(line.strip()) for line in f]

    with open(transcript_manifest_file, "r") as f:
        transcript_manifest = [line.strip() for line in f]

    logger.info("Chunking audio and transcript files")
    start = time.time()
    with multiprocessing.Pool() as pool:
        segments_group = list(
            tqdm(
                pool.imap_unordered(
                    parallel_chunk_transcript,
                    zip(
                        transcript_data,
                        repeat(transcript_manifest),
                        repeat(log_dir),
                        repeat(in_memory),
                    ),
                ),
                total=len(transcript_data),
            )
        )

    logger.info(segments_group[:5])
    segments_group = [group for group in segments_group if group is not None]
    logger.info(f"{segments_group[:5]=}")
    logger.info(f"Time taken to segment: {(time.time() - start) / 60} minutes")

    # Write the data to tar files
    logger.info("Writing data to tar files")
    start = time.time()
    segments_list = list(chain(*segments_group))
    logger.info(f"{segments_list[:5]=}")

    output_path = f"{output_dir}/shard_seg_{shard}.jsonl.gz"
    if subsample:
        rng = np.random.default_rng(subsample_seed)
        subsampled_segments_list = rng.choice(
            segments_list, size=subsample_size, replace=False
        )
        segments_list = subsampled_segments_list

        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            for segment in segments_list:
                f.write(json.dumps(segment) + "\n")
    else:
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            for segment in segments_list:
                f.write(json.dumps(segment) + "\n")

    return output_path, log_dir, len(segments_list)


def main(
    source_dir: str,
    manifest_dir: str,
    log_dir: str,
    output_dir: str,
    subsample: bool = False,
    subsample_size: int = 0,
    subsample_seed: int = 42,
    in_memory: bool = True,
):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    shard_jsonls = glob.glob(f"{source_dir}/*.jsonl.gz")
    get_shard = lambda shard_jsonl: shard_jsonl.split("_")[-1].split(".")[0]
    segment_count = 0
    for shard_jsonl in shard_jsonls:
        logger.info(f"Processing {shard_jsonl}")
        output_path, log_dir, segment_count = preprocess_jsonl(
            json_file=shard_jsonl,
            shard=get_shard(shard_jsonl),
            transcript_manifest_file=f"{manifest_dir}/{get_shard(shard_jsonl)}.txt",
            log_dir=log_dir,
            output_dir=output_dir,
            subsample=subsample,
            subsample_size=subsample_size,
            subsample_seed=subsample_seed,
            in_memory=in_memory,
        )
        logger.info(f"Segmented {segment_count} samples")
        segment_count += 0

    logger.info(f"Total segment count: {segment_count}")


if __name__ == "__main__":
    Fire(main)
