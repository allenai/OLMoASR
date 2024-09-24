import tarfile
import numpy as np
import multiprocessing
from tqdm import tqdm
import glob
import io
from fire import Fire
from typing import List, Literal, Optional
from itertools import repeat

def check_tar(tar_path: str, check: Optional[Literal["count"]]):
    try:
        with tarfile.open(tar_path, "r") as tar:
            members = tar.getmembers()
            if len(members) == 0:
                print(f"{tar_path} is empty")
                return ("empty", tar_path)

            if check == "count":
                if len(members) % 2 != 0:
                    print(f"{tar_path} is missing a .npy/.srt file")
                    return ("missing", tar_path, len(members))
                else:
                    return (len(members) // 2, tar_path)
            else:
                npy_members = [m for m in members if m.name.endswith(".npy")]
                f = tar.extractfile(npy_members[0])
                npy_data = f.read()
                npy_array = np.load(io.BytesIO(npy_data))
                if npy_array.dtype == np.float32:
                    print(f"{tar_path} is float32")
                    return ("float32", tar_path)
    except:
        print(f"{tar_path} is corrupted")
        return ("corrupted", tar_path)

def parallel_check_tar(args):
    return check_tar(*args)
    

def main(tar_paths_str: str, output_file: str, check: Optional[Literal["count"]] = None):
    tar_paths = tar_paths_str.split(",")
    with multiprocessing.Pool() as pool:
        result = list(
            tqdm(
                pool.imap_unordered(
                    parallel_check_tar,
                    zip(tar_paths, repeat(check)),
                ),
                total=len(tar_paths),
            )
        )

    with open(output_file, "a") as f:
        for r in result:
            if r:
                f.write(f"{r}\n")

if __name__ == "__main__":
    Fire(main)
