import os
import multiprocessing
from tqdm import tqdm
import requests
from typing import Optional
from fire import Fire

def download(index: str) -> None:
    url = f"https://mattd-public.s3.us-west-2.amazonaws.com/whisper/{index}.tar.gz"
    file_path = f"/mmfs1/gscratch/efml/hvn2002/ow_440K_tar/{index}.tar.gz"
    temp_file_path = f"{file_path}.tmp"
    
    # Skip if the file already exists
    if os.path.exists(file_path):
        return
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Download the file in chunks and save it to a temporary file
        with open(temp_file_path, "wb") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
        
        # Rename the temporary file to the final file name
        os.rename(temp_file_path, file_path)
    except Exception as e:
        print(f"Failed to download {index}: {e}")

def main(tar_index_str: Optional[str]):
    if tar_index_str is None:
        tar_index = [f"{i:08}" for i in range(2, 2449)]
    else:
        tar_index = tar_index_str.split(",")

    with multiprocessing.Pool() as pool:
        list(tqdm(pool.imap_unordered(download, tar_index), total=len(tar_index)))

if __name__ == "__main__":
    Fire(main)
    
