import multiprocessing
from tqdm import tqdm
import glob
import os
import shutil
from fire import Fire

def move_dirs(src_dest):
    src, dest = src_dest
    if os.path.exists(src):
        os.makedirs(dest, exist_ok=True)
        shutil.move(src, dest)

def main(paths_file: str, dry_run: bool):
    with open(paths_file, "r") as f:
        src_dest_list = [line.strip().split("\t") for line in f]
   
    print(src_dest_list[:5])

    if not dry_run:
        with multiprocessing.Pool() as pool:
            res = list(tqdm(pool.imap_unordered(move_dirs, src_dest_list), total=len(src_dest_list)))

if __name__ == "__main__":
    Fire(main)