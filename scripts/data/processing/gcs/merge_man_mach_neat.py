import json
from tqdm import tqdm
import multiprocessing
from itertools import chain, repeat
import gzip
import os
from google.cloud import storage
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = "AKIASHLPW4FE63DTIAPD"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
from moto3.queue_manager import QueueManager
import subprocess
from typing import Dict, Optional
import shutil

# Initialize the GCS client
client = storage.Client()

def get_id_to_shard(id_langs_batch):
    batch_idx = id_langs_batch[1]
    id_langs = id_langs_batch[0]
    return [(id_lang[0], batch_idx) for id_lang in id_langs]

def extract_jsonl_data(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line.strip()) for line in f]
    return data

def download_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def upload_to_gcs(bucket_name: str, source_file_name: str, destination_blob_name: str):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def modify_dict(shard_dict: Dict):
    video_id = shard_dict["subtitle_file"].split("/")[5]
    if video_id in id_to_machgen_content:
        shard_dict["mach_content"] = id_to_machgen_content[video_id]
    else:
        shard_dict["mach_content"] = ""
    shard_dict["id"] = video_id
    return shard_dict

def parallel_download_from_gcs(args):
    return download_from_gcs(*args)

def main():
    qm = QueueManager("ow_gen_jsonl_2")
    while True:
        try:
            message, item = qm.get_next(visibility_timeout=60 * 60)
            shard_jsonl = item["shard_jsonl"]
            print(f"Downloading {shard_jsonl}")
            
            download_from_gcs(
                bucket_name="ow-seg",
                source_blob_name=f"jsonl_no_mach/{shard_jsonl}",
                destination_file_name=f"{shard_jsonl}",
            )

            with gzip.open(f"{shard_jsonl}", "rt") as f:
                shard_dicts = [json.loads(line.strip()) for line in f]
            print(f"{len(shard_dicts)=}")

            print("Downloading all machgen_jsonls")
            os.makedirs("machgen_jsonls", exist_ok=True)
            # Determine start index and batch of JSONL files
            if int(shard_jsonl.split("_")[-1].split(".")[0]) > 12449:
                start_machgen_batch_idxs = int(shard_jsonl.split("_")[-1].split(".")[0]) + ((37164 - 12449) + 3 * (int(shard_jsonl.split("_")[-1].split(".")[0]) - 12449))
                machgen_jsonls = [f"{i:08}.jsonl" for i in range(start_machgen_batch_idxs, start_machgen_batch_idxs + 4)]
            else:
                start_machgen_batch_idxs = int(shard_jsonl.split("_")[-1].split(".")[0]) + (3 * int(shard_jsonl.split("_")[-1].split(".")[0]))
                machgen_jsonls = [f"{i:08}.jsonl" for i in range(start_machgen_batch_idxs, start_machgen_batch_idxs + 4)]

            # Sequential download using a for loop
            for jsonl in tqdm(machgen_jsonls, total=len(machgen_jsonls)):
                parallel_download_from_gcs((
                    "ow-download",
                    f"jsonl_machgen_1/{jsonl}",
                    f"machgen_jsonls/{jsonl}"
                ))
                
            # if int(shard_jsonl.split("_")[-1]) > 12449:
            #     start_machgen_batch_idxs = int(shard_jsonl.split("_")[-1]) + ((37164 - 12449) + 3 * (int(shard_jsonl.split("_")[-1]) - 12449))
            #     machgen_jsonls = [f"{i:08}.jsonl" for i in range(start_machgen_batch_idxs, start_machgen_batch_idxs + 4)]
            # else:
            #     start_machgen_batch_idxs = int(shard_jsonl.split("_")[-1]) + (3 * int(shard_jsonl.split("_")[-1]))
            #     machgen_jsonls = [f"{i:08}.jsonl" for i in range(start_machgen_batch_idxs, start_machgen_batch_idxs + 4)]

            # with multiprocessing.Pool() as pool:
            #     res = list(
            #         tqdm(
            #             pool.imap_unordered(
            #                 parallel_download_from_gcs,
            #                 zip(
            #                     repeat("ow-download"),
            #                     [f"jsonl_machgen_1/{p}" for p in machgen_jsonls],
            #                     [f"machgen_jsonls/{p}" for p in machgen_jsonls],
            #                 ),
            #             ),
            #             total=len(machgen_jsonls),
            #         )
            #     )

            machgen_paths = [f"machgen_jsonls/{p}" for p in machgen_jsonls if os.path.exists(f"machgen_jsonls/{p}")]
            print(f"{len(machgen_paths)=}")
            print(f"{machgen_paths[0]=}")

            # print("Ensuring all files are downloaded")
            # for p in machgen_paths:
            #     if not os.path.exists(p):
            #         download_from_gcs(
            #             bucket_name="allennlp-mattj",
            #             source_blob_name=f"openwhisper/pretraining_data/jsonl_mach/{os.path.basename(p)}",
            #             destination_file_name=p,
            #         )

            print("Extracting data from machgen_jsonls")
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

            with gzip.open(shard_jsonl, "wt", encoding="utf-8") as f:
                for d in shard_dicts:
                    f.write(json.dumps(d) + "\n")

            upload_to_gcs(
                bucket_name="ow-seg",
                source_file_name=shard_jsonl,
                destination_blob_name=f"fulldata_2/{shard_jsonl}",
            )
            qm.delete(message)
            shutil.rmtree("machgen_jsonls")
            os.remove(shard_jsonl)
        except IndexError:
            print("No more items in queue to process")
            break

if __name__ == "__main__":
    main()
