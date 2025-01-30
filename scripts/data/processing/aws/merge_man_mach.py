import json
from tqdm import tqdm
import multiprocessing
from itertools import chain, repeat
import gzip
import os

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = "AKIASHLPW4FE63DTIAPD"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
from moto3.queue_manager import QueueManager
import subprocess
from typing import Dict, Optional
import shutil


def get_id_to_shard(id_langs_batch):
    batch_idx = id_langs_batch[1]
    id_langs = id_langs_batch[0]
    return [(id_lang[0], batch_idx) for id_lang in id_langs]


def extract_jsonl_data(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line.strip()) for line in f]

    return data


def download_from_s3(
    bucket: str, bucket_prefix: str, file_name: str, output_dir: Optional[str]
):
    cmd = [
        "aws",
        "s3",
        "cp",
        f"s3://{bucket}/{bucket_prefix}/{file_name}",
        "." if output_dir is None else output_dir,
        "--quiet",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)


def upload_to_s3(file_path: str, bucket: str, bucket_prefix: str):
    cmd = [
        "aws",
        "s3",
        "cp",
        file_path,
        f"s3://{bucket}/{bucket_prefix}/",
        "--quiet",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)


def modify_dict(shard_dict: Dict):
    video_id = shard_dict["subtitle_file"].split("/")[5]
    if video_id in id_to_machgen_content:
        shard_dict["mach_content"] = id_to_machgen_content[video_id]
    else:
        shard_dict["mach_content"] = ""
    shard_dict["id"] = video_id

    return shard_dict


def parallel_download_from_s3(args):
    return download_from_s3(*args)


def main():
    qm = QueueManager("ow_gen_jsonl")

    if not os.path.exists("/tmp/id_to_machgen_shard.json.gz"):
        print("Downloading id_to_machgen_shard.json.gz")
        download_from_s3(
            bucket="allennlp-mattj",
            bucket_prefix="openwhisper/pretraining_data",
            file_name="id_to_machgen_shard.json.gz",
            output_dir="/tmp",
        )

    print("Loading id_to_machgen_shard.json.gz")
    with gzip.open(
        "/tmp/id_to_machgen_shard.json.gz", "rt", encoding="utf-8"
    ) as gz_file:
        id_to_machgen_shard = json.load(gz_file)

    while True:
        try:
            message, item = qm.get_next(visibility_timeout=60 * 60)
            shard_jsonl = item["shard_jsonl"]
            print(f"Downloading {shard_jsonl}")

            download_from_s3(
                bucket="allennlp-mattj",
                bucket_prefix="openwhisper/pretraining_data/jsonl_no_mach",
                file_name=shard_jsonl,
                output_dir="/tmp",
            )

            with gzip.open("/tmp/" + shard_jsonl, "rt") as f:
                shard_dicts = [json.loads(line.strip()) for line in f]
            print(f"{len(shard_dicts)=}")

            video_ids = [d["subtitle_file"].split("/")[5] for d in shard_dicts]
            print(f"{len(video_ids)=}")
            print(f"{video_ids[0]=}")

            id_to_machgen_shard_mini = {
                video_id: (
                    f"{id_to_machgen_shard[video_id]:08}.jsonl"
                    if video_id in id_to_machgen_shard
                    else ""
                )
                for video_id in video_ids
            }
            machgen_jsonls = list(set(id_to_machgen_shard_mini.values()))
            print(f"{len(machgen_jsonls)=}")
            print(f"{machgen_jsonls[0]=}")

            print("Downloading all machgen_jsonls")
            os.makedirs("/tmp/machgen_jsonls", exist_ok=True)
            with multiprocessing.Pool() as pool:
                res = list(
                    tqdm(
                        pool.imap_unordered(
                            parallel_download_from_s3,
                            zip(
                                repeat("allennlp-mattj"),
                                repeat("openwhisper/pretraining_data/jsonl_mach"),
                                machgen_jsonls,
                                repeat("/tmp/machgen_jsonls"),
                            ),
                        ),
                        total=len(machgen_jsonls),
                    )
                )

            machgen_paths = [f"/tmp/machgen_jsonls/{p}" for p in machgen_jsonls]
            print(f"{len(machgen_paths)=}")
            print(f"{machgen_paths[0]=}")

            print("Ensuriung all files are downloaded")
            for p in machgen_paths:
                if not os.path.exists(p):
                    download_from_s3(
                        bucket="allennlp-mattj",
                        bucket_prefix="openwhisper/pretraining_data/jsonl_mach",
                        file_name=os.path.basename(p),
                        output_dir="/tmp/machgen_jsonls",
                    )

            print("Extracting data from machgen_jsonls")
            # Loop through each file
            with multiprocessing.Pool() as pool:
                big_list = list(
                    chain(
                        *tqdm(
                            pool.imap_unordered(extract_jsonl_data, machgen_paths),
                            total=len(machgen_paths),
                        )
                    )
                )
            print(f"{len(big_list)=}")

            global id_to_machgen_content
            id_to_machgen_content = {d["id"]: d["mach_content"] for d in big_list}

            print("Modifying shard_dicts")
            with multiprocessing.Pool() as pool:
                shard_dicts = list(
                    tqdm(
                        pool.imap_unordered(modify_dict, shard_dicts),
                        total=len(shard_dicts),
                    )
                )
            print(f"{len(shard_dicts)=}")

            assert len(shard_dicts) == len(video_ids)

            with gzip.open(shard_jsonl, "wt", encoding="utf-8") as f:
                for d in shard_dicts:
                    f.write(json.dumps(d) + "\n")

            upload_to_s3(
                shard_jsonl,
                "allennlp-mattj",
                "openwhisper/pretraining_data/raw_full_jan25",
            )
            qm.delete(message)
            shutil.rmtree("/tmp/machgen_jsonls")
            os.remove(shard_jsonl)
            os.remove("/tmp/" + shard_jsonl)
        except IndexError:
            print("No more items in queue to process")
            break


if __name__ == "__main__":
    main()
