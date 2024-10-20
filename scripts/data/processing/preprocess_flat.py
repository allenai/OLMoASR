import os
import tarfile
import shutil
import glob
from tqdm import tqdm
from typing import List, Tuple
from open_whisper.preprocess import parallel_chunk_audio_transcript
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
    handlers=[logging.FileHandler("download.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def write_to_disk(segment: Tuple[str, str, str, np.ndarray]) -> None:
    t_output_file, transcript_string, a_output_file, audio_arr = segment
    # Write transcript
    with open(t_output_file, "w") as f:
        f.write(transcript_string)
    # Write audio
    np.save(a_output_file, audio_arr)


def preprocess(
    output_dir: str,
    source_dir: str,
    log_dir: str,
    preproc_fail_dir: str,
    jobs_batch_size: int,
    start_shard_idx: int,
    end_shard_idx: int,
    in_memory: bool = True,
) -> None:
    job_batch_idx = int(os.getenv("JOB_BATCH_IDX"))
    job_idx = int(os.getenv("BEAKER_REPLICA_RANK"))
    data_shard_path = sorted(glob.glob(source_dir + "/*"))[
        start_shard_idx:end_shard_idx + 1
    ][job_idx + (job_batch_idx * jobs_batch_size)]

    output_dir = os.path.join(output_dir, f"{data_shard_path.split('/')[-1]}")
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, f"{data_shard_path.split('/')[-1]}")
    os.makedirs(log_dir, exist_ok=True)
    preproc_fail_dir = os.path.join(preproc_fail_dir, f"{data_shard_path.split('/')[-1]}")
    os.makedirs(preproc_fail_dir, exist_ok=True)
    logger.info(f"{output_dir=}, {log_dir=}, {preproc_fail_dir=}")

    logger.info(f"Preprocessing {data_shard_path}")
    audio_files = sorted(glob.glob(data_shard_path + "/*/*.m4a"))
    transcript_files = sorted(glob.glob(data_shard_path + "/*/*.vtt"))

    logger.info(f"{len(audio_files)} audio files")
    logger.info(f"{len(transcript_files)} transcript files")

    if len(audio_files) != len(transcript_files):
        logger.info(f"Uneven number of audio and transcript files in {data_shard_path}")
        with open(
            os.path.join(
                log_dir, f"uneven_data_shards.txt"
            ),
            "a",
        ) as f:
            f.write(f"{len(audio_files)}\t{len(transcript_files)}\n")
        return None

    # Chunk the audio and transcript files
    logger.info("Chunking audio and transcript files")
    start = time.time()
    with multiprocessing.Pool(multiprocessing.cpu_count() * 7) as pool:
        segments_group = list(
            tqdm(
                pool.imap_unordered(
                    parallel_chunk_audio_transcript,
                    zip(
                        transcript_files,
                        audio_files,
                        repeat(output_dir),
                        repeat(log_dir),
                        repeat(preproc_fail_dir),
                        repeat(in_memory),
                    ),
                ),
                total=len(transcript_files),
            )
        )
    logger.info(segments_group[:5])
    # segments group is [[(t_output_file, transcript_string, a_output_file, audio_arr), ...], ...]
    # where each inner list is a group of segments from one audio-transcript file, and each tuple is a segment
    segments_group = [group for group in segments_group if group is not None]
    logger.info(f"Time taken to segment: {(time.time() - start) / 60} minutes")

    # Write the data to tar files
    # logger.info("Writing data to disk")
    # segments_list = list(chain(*segments_group))

    # start = time.time()
    # with multiprocessing.Pool() as pool:
    #     res = list(
    #         tqdm(
    #             pool.imap_unordered(
    #                 write_to_disk,
    #                 segments_list,
    #             ),
    #             total=len(segments_list),
    #         )
    #     )
    # logger.info(f"Time taken to write to disk: {(time.time() - start) / 60} minutes")

    logger.info(f"Completed processing data shard {data_shard_path}")


if __name__ == "__main__":
    Fire(preprocess)
