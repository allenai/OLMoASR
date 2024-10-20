import multiprocessing
from tqdm import tqdm
import glob
import os
import tarfile
from fire import Fire
from itertools import repeat
from typing import Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.FileHandler("download.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def unarchive_tar_file(tar_files_dir_idx: Tuple, output_dir):
    """
    Unarchives a single tar file.

    Args:
        tar_file (str): Path to the tar file.

    Returns:
        str: Message indicating the completion of the task.
    """
    output_dir = os.path.join(output_dir, tar_files_dir_idx[1])
    os.makedirs(output_dir, exist_ok=True)

    for tar_file in tar_files_dir_idx[0]:
        with tarfile.open(tar_file, "r") as tar:
            tar.extractall(path=output_dir)


def unarchive_tar_file_parallel(args):
    return unarchive_tar_file(*args)


def unarchive_tar(
    source_dir: str,
    base_output_dir: str,
    start_dir_idx: int,
    batch_size: int,
    group_by: int,
):
    batch_idx = int(os.getenv("BEAKER_REPLICA_RANK"))
    batch_tar = sorted(glob.glob(os.path.join(source_dir, "*")))[
        batch_idx * batch_size : (batch_idx * batch_size) + batch_size
    ]
    start_dir_idx = start_dir_idx + ((batch_size // group_by) * batch_idx)
    tar_files_dir_idx = [
        (batch_tar[i : i + group_by], f"{(start_dir_idx + (i // group_by)):05}")
        for i in range(0, len(batch_tar), group_by)
    ]

    print(f"{batch_idx=}")
    print(f"{batch_tar=}")
    print(f"{start_dir_idx=}")
    print(f"{tar_files_dir_idx=}")

    with multiprocessing.Pool() as pool:
        res = list(
            tqdm(
                pool.imap_unordered(
                    unarchive_tar_file_parallel,
                    zip(
                        tar_files_dir_idx,
                        repeat(base_output_dir),
                    ),
                ),
                total=len(tar_files_dir_idx),
            )
        )


if __name__ == "__main__":
    Fire({"unarchive_tar": unarchive_tar, "unarchive_tar_file": unarchive_tar_file})
