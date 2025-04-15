import os
import gc
import tarfile
import shutil
import gzip
import json
import glob
from tqdm import tqdm
from typing import List, Tuple, Union, Dict, Optional
from open_whisper.preprocess import chunk_local, chunk_transcript_only
import time
import multiprocessing
from fire import Fire
from tempfile import TemporaryDirectory, NamedTemporaryFile
from itertools import repeat, chain
import numpy as np
from io import BytesIO
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def write_to_disk(
    segment: Union[Tuple[str, np.ndarray], Tuple[str, str, str, np.ndarray]],
) -> None:
    if len(segment) == 2:
        a_output_file, audio_arr = segment
    else:
        t_output_file, transcript_string, a_output_file, audio_arr = segment

    os.makedirs(os.path.dirname(a_output_file), exist_ok=True)

    if len(segment) == 4:
        # Write transcript
        with open(t_output_file, "w") as f:
            f.write(transcript_string)
    # Write audio
    np.save(a_output_file, audio_arr)


def parallel_chunk_local(args):
    return chunk_local(*args)


def parallel_chunk_transcript_only(args):
    return chunk_transcript_only(*args)


def preprocess(
    output_dir: str,
    source_dir: str,
    log_dir: str,
    job_batch_size: int,
    start_shard_idx: int,
    missing_pair_dir: Optional[str] = None,
    manifest_dir: Optional[str] = None,
    audio_only: bool = False,
    transcript_only: bool = False,
    in_memory: bool = True,
) -> None:
    job_idx = int(os.getenv("BEAKER_REPLICA_RANK", 0))
    job_start_shard_idx = start_shard_idx + (job_idx * job_batch_size)
    job_end_shard_idx = start_shard_idx + ((job_idx + 1) * job_batch_size)
    print(f"{job_start_shard_idx=}")
    print(f"{job_end_shard_idx=}")
    data_shard_paths = sorted(glob.glob(source_dir + "/*"))[
        job_start_shard_idx : job_end_shard_idx + 1
    ]
    print(f"{data_shard_paths=}")

    for data_shard_path in data_shard_paths:
        data_shard_idx = ""
        segment_output_dir = output_dir
        if data_shard_path.endswith(".jsonl.gz"):
            data_shard_idx = os.path.basename(data_shard_path).split("_")[-1].split(".")[0]

            if transcript_only is False:
                segment_output_dir = os.path.join(
                    output_dir,
                    f"{data_shard_idx}",
                )
        else:
            data_shard_idx = data_shard_path.split("/")[-1]
            segment_output_dir = os.path.join(output_dir, f"{data_shard_idx}")

        print(f"{data_shard_path=}, {data_shard_idx=}, {segment_output_dir=}")

        os.makedirs(segment_output_dir, exist_ok=True)

        if not data_shard_path.endswith(".jsonl.gz"):
            # dealing w/ missing pairs
            missing_pair_dir = os.path.join(
                missing_pair_dir, f"{data_shard_path.split('/')[-1]}"
            )
            os.makedirs(missing_pair_dir, exist_ok=True)
            logger.info(f"{segment_output_dir=}, {missing_pair_dir=}")

            logger.info(f"Preprocessing {data_shard_path}")
            audio_files = sorted(glob.glob(data_shard_path + "/*/*.m4a"))
            transcript_files = sorted(glob.glob(data_shard_path + "/*/*.*t"))

            logger.info(f"{len(audio_files)} audio files")
            logger.info(f"{len(transcript_files)} transcript files")

            if len(audio_files) != len(transcript_files):
                logger.info(
                    f"Uneven number of audio and transcript files in {data_shard_path}"
                )
                if len(audio_files) > len(transcript_files):
                    missing_pairs = [
                        "/".join(p.split("/")[:6])
                        for p in (
                            set([p.split(".")[0] for p in audio_files])
                            - set([p.split(".")[0] for p in transcript_files])
                        )
                    ]
                    new_paths = [shutil.move(d, missing_pair_dir) for d in missing_pairs]
                elif len(audio_files) < len(transcript_files):
                    missing_pairs = [
                        "/".join(p.split("/")[:6])
                        for p in (
                            set([p.split(".")[0] for p in transcript_files])
                            - set([p.split(".")[0] for p in audio_files])
                        )
                    ]
                    new_paths = [shutil.move(d, missing_pair_dir) for d in missing_pairs]

                logger.info(f"{new_paths[:5]=}")

                audio_files = sorted(glob.glob(data_shard_path + "/*/*.m4a"))
                transcript_files = sorted(glob.glob(data_shard_path + "/*/*.*t"))
        else:
            if transcript_only is False:
                with gzip.open(data_shard_path, "rt") as f:
                    data = [json.loads(line.strip()) for line in f]

                audio_files = sorted([d["audio_file"] for d in data])
                transcript_files = sorted([d["subtitle_file"] for d in data])

                assert len(audio_files) == len(
                    transcript_files
                ), f"Uneven number of audio and transcript files in {data_shard_path}"
            else:
                with gzip.open(data_shard_path, "rt") as f:
                    data = [json.loads(line.strip()) for line in f]

                manifest_file = os.path.join(manifest_dir, f"{data_shard_idx}.txt")
                with open(manifest_file, "r") as f:
                    transcript_manifest = [line.strip() for line in f]

        # debug
        # print(f"{audio_files[0]=}")
        # print(f"{transcript_files[0]=}")
        # chunk_local(transcript_file=transcript_files[0], audio_file=audio_files[0], output_dir=segment_output_dir, audio_only=audio_only, transcript_only=transcript_only, in_memory=in_memory)
        # chunk_transcript_only(data[10], transcript_manifest, segment_output_dir)

        # Chunk data
        logger.info("Chunking data")
        start = time.time()
        if transcript_only is False:
            with multiprocessing.Pool(multiprocessing.cpu_count() * 7) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(
                            parallel_chunk_local,
                            zip(
                                transcript_files,
                                audio_files,
                                repeat(segment_output_dir),
                                repeat(audio_only),
                                repeat(transcript_only),
                                repeat(in_memory),
                            ),
                        ),
                        total=len(transcript_files),
                    )
                )
        else:
            with multiprocessing.Pool(multiprocessing.cpu_count() * 7) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(
                            parallel_chunk_transcript_only,
                            zip(data, repeat(transcript_manifest), repeat(segment_output_dir)),
                        ),
                        total=len(data),
                    )
                )

        (
            segments_group,
            over_30_line_segment_count,
            bad_text_segment_count,
            over_ctx_len_segment_count,
            faulty_audio_segment_count,
            faulty_transcript_count,
            failed_transcript_count,
        ) = zip(*results)

        logger.info(f"{segments_group[:5]=}")
        # segments group is [[(t_output_file, transcript_string, a_output_file, audio_arr), ...], ...] or [[(a_output_file, audio_arr), ...], ...]
        # where each inner list is a group of segments from one audio-transcript file, and each tuple is a segment
        segments_group = [group for group in segments_group if group is not None]
        logger.info(f"Time taken to segment: {(time.time() - start) / 60} minutes")

        # Write the data to tar files
        logger.info("Writing data to disk")
        segments_list = list(chain(*segments_group))

        start = time.time()
        if transcript_only is False:
            with multiprocessing.Pool() as pool:
                _ = list(
                    tqdm(
                        pool.imap_unordered(
                            write_to_disk,
                            segments_list,
                        ),
                        total=len(segments_list),
                    )
                )
        else:
            with gzip.open(
                os.path.join(segment_output_dir, f"{data_shard_idx}.jsonl.gz"), "wt"
            ) as f:
                [f.write(f"{json.dumps(segment)}\n") for segment in segments_list]
        logger.info(f"Time taken to write to disk: {(time.time() - start) / 60} minutes")

        logger.info(f"Completed processing data shard {data_shard_path}")

        # logging process stats
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/{data_shard_idx}.json", "w") as f:
            log_data = {
                "over_30_line_segment_count": sum(over_30_line_segment_count),
                "bad_text_segment_count": sum(bad_text_segment_count),
                "over_ctx_len_segment_count": sum(over_ctx_len_segment_count),
                "faulty_audio_segment_count": sum(faulty_audio_segment_count),
                "faulty_transcript_count": sum(faulty_transcript_count),
                "failed_transcript_count": sum(failed_transcript_count),
            }
            f.write(json.dumps(log_data, indent=4))

        # writing manifest file
        if transcript_only is False:
            os.makedirs(manifest_dir, exist_ok=True)
            with open(f"{manifest_dir}/{data_shard_idx}.txt", "w") as f:
                [
                    f.write(f"{'/'.join(segment[-2].split('/')[-2:])}\n")
                    for segment in segments_list
                ]

        del audio_files
        del transcript_files
        del results
        del segments_group
        del segments_list
        
        gc.collect()

if __name__ == "__main__":
    Fire(preprocess)
    # preprocess(
    #     output_dir="temp",
    #     source_dir="intermediate_data/text_heurs_1_manmach_0.9_editdist_0.28_feb_25",
    #     log_dir="temp_logs",
    #     jobs_batch_size=1,
    #     start_shard_idx=0,
    #     end_shard_idx=0,
    #     audio_only=True,
    #     transcript_only=False,
    #     in_memory=True,
    # )
    # preprocess(
    #     output_dir="temp_jsonl",
    #     source_dir="intermediate_data/text_heurs_1_manmach_0.9_editdist_0.28_feb_25",
    #     log_dir="temp_jsonl_logs",
    #     jobs_batch_size=1,
    #     start_shard_idx=0,
    #     end_shard_idx=0,
    #     manifest_dir="temp_logs",
    #     transcript_only=True,
    #     in_memory=True,
    # )
