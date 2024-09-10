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
from itertools import repeat
import numpy as np

TARS_PATH = "/mmfs1/gscratch/efml/hvn2002/ow_440K_wds"

def split_list(data_shard_idx: int, lst: List, n: int) -> List[Tuple]:
    rng = np.random.default_rng(42)
    rng.shuffle(lst)
    start_shard_idx = data_shard_idx + ((n - 1) * data_shard_idx)
    # Calculate size of each portion
    k, m = divmod(len(lst), n)
    # Create each portion and append to the result
    return [
        (lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)], start_shard_idx + i)
        for i in range(n)
    ]


def write_to_tar(
    job_batch_idx: int, job_idx: int, data_shard_idx: int, segment_shards: List[Tuple]
) -> None:
    segments, shard_idx = segment_shards
    tar_name = f"{shard_idx:06}.tar"

    with NamedTemporaryFile(delete=False, suffix=".tar") as temp_tar:
        tar_path = temp_tar.name

        with tarfile.open(tar_path, "w") as tar:
            for path in tqdm(segments, total=len(segments)):
                tar.add(path, arcname=path.split("/")[-2])
            with open("logs/data/preprocess/num_files.txt", "a") as f:
                f.write(f"{shard_idx}: {len(segments)}\n") 

    shutil.move(tar_path, os.path.join(TARS_PATH, tar_name))
def preprocess(
    job_batch_idx: int, job_idx: int, data_shard_path: str, num_output_shards: int = 30, in_memory: bool = True
) -> None:
    if not os.path.exists(data_shard_path):
        print(f"{data_shard_path} does not exist, do not proceed to preprocess to webdataset shards")
        return None
    
    print(f"Preprocessing {data_shard_path} with {num_output_shards} output shards")
    log_dir = "logs/data/preprocess"
    os.makedirs(TARS_PATH, exist_ok=True)
    audio_files = sorted(glob.glob(data_shard_path + "/*/*.m4a"))
    transcript_files = sorted(glob.glob(data_shard_path + "/*/*.srt"))
    data_shard_idx = int(data_shard_path.split("/")[-1])

    print(f"{len(audio_files)} audio files")
    print(f"{len(transcript_files)} transcript files")

    if len(audio_files) != len(transcript_files):
        with open(
            os.path.join(LOG_DIR, f"uneven_data_shards_{job_batch_idx}.txt"), "a"
        ) as f:
            f.write(f"{data_shard_idx}\t{len(audio_files)}\t{len(transcript_files)}\n")
        return None

    with open(
        os.path.join(LOG_DIR, f"processing_data_shards_{job_batch_idx}.txt"), "a"
    ) as f:
        f.write(f"{job_idx}\t{data_shard_path}\n")

    # Chunk the audio and transcript files
    with TemporaryDirectory() as data_shard_temp_dir:
        print("Chunking audio and transcript files")
        start = time.time()
        with multiprocessing.Pool(multiprocessing.cpu_count() * 7) as pool:
            segments_group = list(
                tqdm(
                    pool.imap_unordered(
                        parallel_chunk_audio_transcript,
                        zip(transcript_files, audio_files, repeat(data_shard_temp_dir), repeat(in_memory)),
                    ),
                    total=len(transcript_files),
                )
            )
        print(segments_group[:5])
        # segments group is [[(t_output_file, transcript_string, a_output_file, audio_arr), ...], ...] 
        # where each inner list is a group of segments from one audio-transcript file, and each tuple is a segment
        segments_group = [group for group in segments_group if group is not None]
        print(f"Time taken to segment: {time.time() - start}")

        # Write the data to tar files
        print("Writing data to tar files")
        start = time.time()
        data_shard_idx = int(data_shard_path.split("/")[-1])
        segments_list = list(chain(*segments_group))
        segment_shards = split_list(
            data_shard_idx=data_shard_idx, lst=segment_files, n=num_output_shards
        )

        with multiprocessing.Pool() as pool:
            out = list(
                tqdm(
                    pool.imap_unordered(write_to_tar, segment_shards),
                    total=len(segment_shards),
                )
            )
        print(
            f"Time taken to write to {num_output_shards} tar files: {time.time() - start}"
        )
    
    print(f"Completed processing data shard {data_shard_path}, cleaning up now")
    start = time.time()
    shutil.rmtree(data_shard_path)
    print(f"Time taken to clean up: {time.time() - start}")

if __name__ == "__main__":
    Fire(preprocess)