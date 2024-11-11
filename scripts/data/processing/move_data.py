import multiprocessing
from tqdm import tqdm
import glob
import os
import shutil
from fire import Fire
from typing import Optional
import numpy as np

def move_dirs(src_dest):
    src, dest = src_dest
    if os.path.exists(src):
        os.makedirs(dest, exist_ok=True)
        shutil.move(src, dest)

def main(paths_file: str, dry_run: bool, batch_size: Optional[int] = None):
    with open(paths_file, "r") as f:
        src_dest_list = [line.strip().split("\t") for line in f]
    
    if batch_size:
        batch_idx = int(os.getenv("BEAKER_REPLICA_RANK"))
        src_dest_list = src_dest_list[batch_size * batch_idx: batch_size * (batch_idx + 1)]
        print(f"{batch_idx=}")
        print(f"{batch_size=}")
        
    print(src_dest_list[:5])
    print(len(src_dest_list))

    if not dry_run:
        with multiprocessing.Pool() as pool:
            res = list(tqdm(pool.imap_unordered(move_dirs, src_dest_list), total=len(src_dest_list)))

if __name__ == "__main__":
    Fire(main)