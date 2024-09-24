import multiprocessing
from tqdm import tqdm
import glob
import os
import tarfile
from fire import Fire
from itertools import repeat


def unarchive_tar_file(tar_file, output_dir, result_path):
    """
    Unarchives a single tar file.

    Args:
        tar_file (str): Path to the tar file.

    Returns:
        str: Message indicating the completion of the task.
    """
    if tar_file.endswith(".tar.gz"):
        output_dir = os.path.join(
            output_dir, f"{int(os.path.basename(tar_file)[:-7]):05}"
        )
        os.makedirs(output_dir, exist_ok=True)

    print(f"Extracting {tar_file} to {output_dir}")

    if "tar" not in tar_file or not os.path.exists(tar_file):
        return None
    
    try:
        with tarfile.open(tar_file, "r") as tar:
            tar.extractall(path=output_dir)

        os.remove(tar_file)

        with open(result_path, "a") as f:
            f.write(f"{tar_file} successful\n")

    except Exception as e:
        with open(result_path, "a") as f:
            f.write(f"{tar_file} failed: {e}\n")


def unarchive_tar_file_parallel(args):
    return unarchive_tar_file(*args)


def unarchive_tar(source_dir, result_path, batch_size, batch_idx):
    source_dir = os.path.join("/".join(source_dir.split("/")[:-1]), f"{(int(source_dir.split('/')[-1]) + (batch_size * batch_idx)):05}")
    print(source_dir)
    tar_files = glob.glob(os.path.join(source_dir, "*"))

    while len(glob.glob(os.path.join(source_dir, "*.tar"))) > 0:
        with multiprocessing.Pool() as pool:
            res = list(
                tqdm(
                    pool.imap_unordered(
                        unarchive_tar_file_parallel,
                        zip(tar_files, repeat(source_dir), repeat(result_path)),
                    ),
                    total=len(tar_files),
                )
            )


if __name__ == "__main__":
    Fire({"unarchive_tar": unarchive_tar, "unarchive_tar_file": unarchive_tar_file})
