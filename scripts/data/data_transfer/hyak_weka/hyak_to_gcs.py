import os
import glob
import subprocess
import multiprocessing
from tqdm import tqdm
from fire import Fire
import json


def upload_file(file_path):
    """
    Function to upload a single file to GCS using gsutil cp command.
    """
    destination_bucket = "gs://huongn-openwhisper/"
    command = ["gsutil", "cp", file_path, destination_bucket]
    try:
        result = subprocess.run(command, check=True)
        with open("logs/data/download/upload_gcs.jsonl", "a") as f:
            f.write(json.dumps({"file_path": file_path, "output": "Complete"}) + "\n")
    except subprocess.CalledProcessError as e:
        with open("logs/data/download/upload_gcs.jsonl", "a") as f:
            f.write(json.dumps({"file_path": file_path, "output": str(e)}) + "\n")
    except Exception as e:
        with open("logs/data/download/upload_gcs.jsonl", "a") as f:
            f.write(json.dumps({"file_path": file_path, "output": str(e)}) + "\n")


def bulk_upload(tar_dir: str, job_index: int, batch_size: int):
    all_tar_files = glob.glob(tar_dir + "/*.tar")
    tar_files_list = all_tar_files[
        job_index * batch_size : (job_index + 1) * batch_size
    ]
    print(tar_files_list[:5])

    print(f"Uploading from {job_index * batch_size} to {(job_index + 1) * batch_size}")

    with multiprocessing.Pool(multiprocessing.cpu_count() * 7) as pool:
        result = list(
            tqdm(
                pool.imap_unordered(upload_file, tar_files_list),
                total=len(tar_files_list),
            )
        )


if __name__ == "__main__":
    Fire({"bulk_upload": bulk_upload, "upload_file": upload_file})
