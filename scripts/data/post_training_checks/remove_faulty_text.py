import tarfile
import glob
import numpy as np
import multiprocessing
from tqdm import tqdm
from typing import List, Tuple
from io import BytesIO
import shutil
import os
from itertools import repeat
from tempfile import NamedTemporaryFile
from fire import Fire

CUSTOM_TEMP_DIR = "/mmfs1/gscratch/efml/hvn2002/temp_dir"
TARS_PATH = "/mmfs1/gscratch/efml/hvn2002/ow_440K_wds"

def extract_files_from_tar(tar_path):
    """Extracts files from a tar file and returns their names and contents."""
    extracted_files = []
    try:
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    file_content = f.read()
                    if member.name.startswith("H5UwQOdI_TQ/00:00:00,100_00:00:29,629"):
                        continue

                    if member.name.endswith(".npy"):
                        npy_data = BytesIO(file_content)
                        extracted_files.append((member.name, npy_data))
                    elif member.name.endswith(".srt"):
                        srt_data = BytesIO(file_content)
                        extracted_files.append((member.name, srt_data))
    except Exception as e:
        print(f"Error processing {tar_path}: {e}")
    return extracted_files


def write_to_tar(segment_shards: List[Tuple]) -> None:
    segments, shard_idx = segment_shards
    tar_name = f"{shard_idx:06}.tar"

    with NamedTemporaryFile(
        delete=False, suffix=".tar", dir=CUSTOM_TEMP_DIR
    ) as temp_tar:
        tar_path = temp_tar.name
        with tarfile.open(tar_path, "w") as tar:
            for paths_data in tqdm(segments, total=len(segments)):
                (t_name, transcript_bytes_io), (a_name, audio_bytes_io) = paths_data
                tarinfo_transcript = tarfile.TarInfo(name=t_name)
                tarinfo_transcript.size = transcript_bytes_io.getbuffer().nbytes
                tar.addfile(tarinfo_transcript, transcript_bytes_io)

                tarinfo_audio = tarfile.TarInfo(name=a_name)
                tarinfo_audio.size = audio_bytes_io.getbuffer().nbytes
                tar.addfile(tarinfo_audio, audio_bytes_io)

    shutil.move(tar_path, os.path.join(TARS_PATH, tar_name))


def parallel_write_to_tar(args):
    return write_to_tar(*args)


def process_tar_files(tar_paths):
    tar_paths_list = tar_paths.split(",")
    print(tar_paths_list[:10])

    print("Extracting files from tar files...")
    with multiprocessing.Pool() as pool:
        results = list(
            tqdm(
                pool.imap_unordered(extract_files_from_tar, tar_paths_list),
                total=len(tar_paths_list),
            )
        )
    
    print("Separating into transcript and audio files...")
    # Flatten the list of lists into a single list of files
    all_extracted_files = [item for sublist in results for item in sublist]
    print(all_extracted_files[:5])
    transcript_files = sorted(
        [item for item in all_extracted_files if item[0].endswith(".srt")]
    )
    audio_files = sorted(
        [item for item in all_extracted_files if item[0].endswith(".npy")]
    )
    transcript_audio = list(zip(transcript_files, audio_files))
    print(transcript_audio[:5])
    with open("transcript_audio.txt", "w") as f:
        for item in transcript_audio:
            f.write(f"{item[0][0]} {item[1][0]}\n")

    # print("Splitting into segments...")
    # segment_shards = split_list(data_shard_idx=1352, lst=transcript_audio, n=30)
    # print(segment_shards[:5])

    # print("Writing to tar files...")
    # with multiprocessing.Pool() as pool:
    #     out = list(
    #         tqdm(
    #             pool.imap_unordered(parallel_write_to_tar, zip(repeat(1352), segment_shards)),
    #             total=len(segment_shards),
    #         )
    #     )

    segment_shards = (transcript_audio, 14533)
    write_to_tar(segment_shards)

if __name__ == "__main__":
    Fire(process_tar_files)