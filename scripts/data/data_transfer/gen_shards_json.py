import base64
import multiprocessing
from tqdm import tqdm
import glob
import json
import time
import os
from fire import Fire
import requests

def encode_tar_file(file_path):
    with open(file_path, "rb") as f:
        # Read the tar file content
        # start = time.time()
        file_content = f.read()
        # Encode the file content to base64
        encoded_content = base64.b64encode(file_content).decode("utf-8")
        # end = time.time()
        print(f"Encoded {file_path}\n")
        # print(f"Time taken to encode: {end - start}\n")
        print(f"Length of encoded content: {len(encoded_content)}\n")
        # print(f"First 1000 characters of encoded content: {encoded_content[:1000]}\n")
    return {"filename": file_path.split("/")[-1], "data": encoded_content}


def bulk_encode(tar_files):
    # Get the list of tar files
    tar_files_list = tar_files.split(",")
    with multiprocessing.Pool() as pool:
        encoded_result = list(
            tqdm(pool.imap_unordered(encode_tar_file, tar_files_list), total=len(tar_files_list))
        )
    print(encoded_result[0].keys())
    print(encoded_result[0]["filename"])
    print(encoded_result[0]["data"][:100])
    # Save the encoded tar files
    # with open(os.path.join(output_dir, f"ow_440K_wds_{array_index}.json"), "a") as f:
    #     json.dump(encoded_result, f)
    
    return encoded_result

def encode_and_upload(tar_files, endpoint):
    encoded_result = bulk_encode(tar_files)
    time.sleep(0.01)
    response = requests.post(endpoint, json=encoded_result, verify=False)
    time.sleep(0.01)
    print(response.status_code)
    print(response.json())

if __name__ == "__main__":
    Fire({"bulk_encode": bulk_encode, "encode_tar_file": encode_tar_file, "encode_and_upload": encode_and_upload})
