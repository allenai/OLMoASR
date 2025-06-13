import subprocess
import glob
import shutil
import tarfile
import os
import numpy as np
import logging
from fire import Fire
from time import time, sleep
import multiprocessing
from tqdm import tqdm
from typing import List, Optional, Literal, Dict, Union, Tuple
import json
from functools import partial
import pandas as pd
import pysrt
import webvtt

os.environ["AWS_DEFAULT_REGION"] = "us-west-1"
os.environ["AWS_ACCESS_KEY_ID"] = "AKIASHLPW4FE63DTIAPD"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
from moto3.queue_manager import QueueManager
from google.cloud import storage, compute_v1
from google.oauth2 import service_account
from itertools import repeat
import traceback


IDENTIFIERS = [
    "unavailable",
    "private",
    "terminated",
    "removed",
    "country",
    "closed",
    "copyright",
    "members",
    "not available",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.FileHandler("download.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# def download(yt_id, output_dir):
#     url = f"https://www.youtube.com/watch?v={yt_id}"

#     # output_path = f"{output_dir}/%(id)s/%(id)s.%(ext)s"
#     output_path = f"{output_dir}/%(id)s.%(ext)s"

#     cmd = [
#         "yt-dlp",
#         url,
#         "--skip-download",
#         "--write-thumbnail",
#         "-o",
#         output_path,
#     ]

#     try:
#         result = subprocess.run(cmd, capture_output=True, text=True)

#         if any(identifier in result.stderr.lower() for identifier in IDENTIFIERS):
#             with open(f"metadata/unavailable.txt", "a") as f:
#                 f.write(f"{yt_id}\n")
#             return "unavailable"
#         elif "not a bot" in result.stderr.lower():
#             with open(f"metadata/blocked_ip.txt", "a") as f:
#                 f.write(f"{yt_id}\t{result.stderr}\t{result.stdout}\n")
#             return "blocked IP"
#         else:
#             thumbnail_files = glob.glob(f"{output_dir}/{yt_id}/*.*")
#             if not any("jpg" in f or "webp" in f or "png" in f for f in thumbnail_files):
#                 with open(f"metadata/unknown.txt", "a") as f:
#                     f.write(f"{yt_id}\tthumbnail\t{result.stderr}\n")
#     except subprocess.CalledProcessError as e:
#         logger.error(f"Error in downloading thumbnail for video {yt_id}: {e.stderr}")
#         return -1

def download(yt_id, output_dir):
    url = f"https://i.ytimg.com/vi/{yt_id}/hqdefault.jpg"
    
    output_path = f"{output_dir}/{yt_id}.jpg"
    cmd = ["wget", url, "-O", output_path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            with open(f"metadata/unavailable.txt", "a") as f:
                f.write(f"{yt_id}\n")
            return "unavailable"
        else:
            return "success"
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in downloading thumbnail for video {yt_id}: {e.stderr}")
        return -1

def parallel_download(args):
    return download(*args)

def main():
    with open("data/1K_samples_ids.txt", "r") as f:
        ids_list = [line.strip() for line in f]
        
    output_dir = "data/thumbnails"
    metadata_dir = "metadata"
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    for vid_id in ids_list:
        print(download(vid_id, output_dir))
        
if __name__ == "__main__":
    main()     
        
        
    
    