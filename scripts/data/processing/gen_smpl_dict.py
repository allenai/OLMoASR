import glob
import os
from typing import Dict, Any, Optional, List
from fire import Fire
from itertools import chain
import logging
import multiprocessing
from tqdm import tqdm
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.FileHandler("main.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

def gen_smpl_dict(segs_dir) -> List:
    if int(segs_dir.split("/")[-2]) > 2448:
        text_files = sorted(glob.glob(segs_dir + "/*.vtt"))
    else:
        text_files = sorted(glob.glob(segs_dir + "/*.srt"))
    npy_files = sorted(glob.glob(segs_dir + "/*.npy"))
    text_npy_samples = list(zip(text_files, npy_files))
    smpl_dicts = []

    for text_fp, npy_fp in text_npy_samples:
        smpl_dict = {"transcript": text_fp, "audio": npy_fp}
        smpl_dicts.append(smpl_dict)

    return smpl_dicts


def main(
    shard_metadata: str,
    samples_dicts_dir: str,
    batch_size: int,
):
    batch_idx = int(os.getenv("BEAKER_REPLICA_RANK"))
    samples_dicts_dir = samples_dicts_dir + f"/{batch_idx:03}"
    os.makedirs(samples_dicts_dir, exist_ok=True)
    logger.info(f"{samples_dicts_dir=}")
    
    with open(shard_metadata, "r") as f:
        shard_metadata = [line.strip() for line in f]
        
    shard_dirs = shard_metadata[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    logger.info(f"{shard_dirs[0]=}, {shard_dirs[-1]=}, {len(shard_dirs)=}")
    data_dirs = list(chain(*[glob.glob(shard_dir + "/*") for shard_dir in shard_dirs]))
    logger.info(f"{data_dirs[:5]=}")

    with multiprocessing.Pool() as pool:
        smpl_dicts = list(tqdm(pool.imap_unordered(gen_smpl_dict, data_dirs), total=len(data_dirs)))

    logger.info("Writing sample dicts to disk")
    with open(f"{samples_dicts_dir}/samples_dicts.jsonl", "w") as f:
        for smpl_dict in smpl_dicts:
            f.write(json.dumps({"sample_dicts": smpl_dict}) + "\n")

if __name__ == "__main__":
    Fire(main)
