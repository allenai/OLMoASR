import subprocess
import os
import logging
from fire import Fire
import multiprocessing
from tqdm import tqdm
from itertools import repeat
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.FileHandler("download.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def download_file(
    file_name: str,
    local_dir: str,
    bucket_name: str,
    bucket_prefix: str,
    log_file: str,
):
    cmd = (
        f"gsutil -m cp -L {log_file} -r gs://{bucket_name}/{bucket_prefix}/{file_name} {local_dir}",
    )
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True)
        logger.info(result.stdout)
        logger.info(result.stderr)
    except Exception as e:
        logger.info(result.stderr)
        logger.error(e)

def download_dir(
    local_dir: str,
    bucket_name: str,
    bucket_prefix: str,
    service_account: str,
    key_file: str,
    log_file: str,
):
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    try:
        cmd = f"gcloud auth activate-service-account '{service_account}' --key-file='{key_file}'"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        logger.info(result.stdout)
        logger.info(result.stderr)
    except Exception as e:
        logger.info(result.stderr)
        logger.error(e)
    
    cmd = (
        f"gsutil -m cp -L {log_file} -r gs://{bucket_name}/{bucket_prefix} {local_dir}"
    )
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True)
        logger.info(result.stdout)
        logger.info(result.stderr)
    except Exception as e:
        logger.info(result.stderr)
        logger.error(e)


def parallel_download_file(args):
    return download_file(*args)


def download_files_set(
    set_file: str,
    local_dir: str,
    bucket_name: str,
    bucket_prefix: str,
    service_account: str,
    key_file: str,
    log_file: str,
    batches: Optional[int] = None,
):
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    try:
        cmd = f"gcloud auth activate-service-account '{service_account}' --key-file='{key_file}'"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        logger.info(result.stdout)
        logger.info(result.stderr)
    except Exception as e:
        logger.info(result.stderr)
        logger.error(e)
    
    with open(set_file, "r") as f:
        file_names = [line.strip() for line in f]
    print(f"{len(file_names)=}")
    print(f"{file_names[-10:]=}")
    
    if batches:
        batch_idx = int(os.getenv("BEAKER_REPLICA_RANK"))
        batch_size = (len(file_names) // batches) + 1
        print(f"{batch_idx=}")
        print(f"{batch_size=}")
        file_names = file_names[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        print(f"{len(file_names)=}")
        print(f"{file_names[-10:]=}")
        
    with multiprocessing.Pool() as pool:
        res = list(
            tqdm(
                pool.imap_unordered(
                    parallel_download_file,
                    zip(
                        file_names,
                        repeat(local_dir),
                        repeat(bucket_name),
                        repeat(bucket_prefix),
                        repeat(log_file),
                    ),
                ),
                total=len(file_names),
            )
        )


def download_files_batch(
    start_dir_idx: int,
    batch_size: int,
    local_dir: str,
    bucket_name: str,
    bucket_prefix: str,
    service_account: str,
    key_file: str,
    log_file: str,
    padding: Optional[int],
    file_ext: Optional[str],
):
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    start_idx = (int(os.getenv("BEAKER_REPLICA_RANK")) * batch_size) + start_dir_idx
    end_idx = start_idx + batch_size
    logger.info(f"Downloading files {start_idx} to {end_idx}")

    try:
        cmd = f"gcloud auth activate-service-account '{service_account}' --key-file='{key_file}'"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        logger.info(result.stdout)
        logger.info(result.stderr)
    except Exception as e:
        logger.info(result.stderr)
        logger.error(e)

    file_names = [
        str(idx).zfill(padding) + f".{file_ext}" for idx in range(start_idx, end_idx)
    ]
    print(f"{file_names[:10]=}")

    with multiprocessing.Pool() as pool:
        res = list(
            tqdm(
                pool.imap_unordered(
                    parallel_download_file,
                    zip(
                        file_names,
                        repeat(local_dir),
                        repeat(bucket_name),
                        repeat(bucket_prefix),
                        repeat(log_file),
                    ),
                ),
                total=batch_size,
            )
        )


if __name__ == "__main__":
    Fire(
        {
            "download_file": download_file,
            "download_files_batch": download_files_batch,
            "download_files_set": download_files_set,
            "download_dir": download_dir,
        }
    )
