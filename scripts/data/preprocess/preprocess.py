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
    start_shard_idx = data_shard_idx + (n * data_shard_idx)
    # Calculate size of each portion
    k, m = divmod(len(lst), n)
    # Create each portion and append to the result
    return [
        (lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)], start_shard_idx + i)
        for i in range(n)
    ]


def write_to_tar(segment_shards: List[Tuple]) -> None:
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
    os.remove(tar_path)


def preprocess(data_shard_path: str, num_output_shards: int) -> None:
    print(f"Preprocessing {data_shard_path} with {num_output_shards} output shards")
    os.makedirs(TARS_PATH, exist_ok=True)
    audio_files = sorted(glob.glob(data_shard_path + "/*/*.m4a"))
    transcript_files = sorted(glob.glob(data_shard_path + "/*/*.srt"))

    print(f"{len(audio_files)} audio files")
    print(f"{len(transcript_files)} transcript files")

    assert len(audio_files) == len(transcript_files)

    # Chunk the audio and transcript files
    print("Chunking audio and transcript files")
    start = time.time()
    with multiprocessing.Pool() as pool:
        out = list(
            tqdm(
                pool.imap_unordered(
                    parallel_chunk_audio_transcript,
                    zip(
                        transcript_files,
                        audio_files,
                    ),
                ),
                total=len(transcript_files),
            )
        )
    print(f"Time taken to segment: {time.time() - start}")

    # Write the data to tar files
    print("Writing data to tar files")
    start = time.time()
    data_shard_idx = int(data_shard_path.split("/")[-1])
    video_id_dirs = glob.glob(data_shard_path + "/*")
    video_id_shards = split_list(
        data_shard_idx=data_shard_idx, lst=video_id_dirs, n=num_output_shards
    )

    with multiprocessing.Pool() as pool:
        out = list(
            tqdm(
                pool.imap_unordered(write_to_tar, video_id_shards),
                total=len(video_id_shards),
            )
        )
    print(
        f"Time taken to write to {num_output_shards} tar files: {time.time() - start}"
    )

if __name__ == "__main__":
    Fire(preprocess)