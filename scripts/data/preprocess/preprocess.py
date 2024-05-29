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

TARS_PATH = "data/tars"


def split_list(data_shard_idx: int, lst: List, n: int) -> List[Tuple]:
    start_shard_idx = data_shard_idx + (5 * (data_shard_idx - 1)) if data_shard_idx > 0 else 0
    # Calculate size of each portion
    k, m = divmod(len(lst), n)
    # Create each portion and append to the result
    return [
        (lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)], start_shard_idx + i)
        for i in range(n)
    ]


def write_to_tar(video_id_shard: List[Tuple]) -> None:
    dirs, shard_idx = video_id_shard
    tar_name = f"{shard_idx:06}.tar"
    tar_path = os.path.join(TARS_PATH, tar_name)

    # Create a tar.gz archive for the group
    with tarfile.open(tar_path, "w") as tar:
        for dir_path in tqdm(dirs, total=len(dirs)):
            tar.add(dir_path, arcname=dir_path.split("/")[-1])


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
    print(f"Time taken to write to {num_output_shards} tar files: {time.time() - start}")


if __name__ == "__main__":
    Fire(preprocess)