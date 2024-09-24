import multiprocessing
from tqdm import tqdm
import os
import shutil
import glob
from itertools import repeat


def check_files(wds_dir, ds_dir, result_path):
    wds_files = set(os.listdir(wds_dir))
    ds_files = set(os.listdir(ds_dir))

    extra_files = ds_files - wds_files

    # with open(result_path, "a") as f:
    if len(extra_files) > 0:
        for file in extra_files:
            shutil.rmtree(os.path.join(ds_dir, file))


def parallel_check_files(args):
    return check_files(*args)


if __name__ == "__main__":
    wds_shards = "/weka/huongn/ow_440K_wds"
    data_shards  = "/weka/huongn/ow_440K_tar"
    wds_dirs = sorted(glob.glob(os.path.join(wds_shards, "*")))
    ds_dirs = sorted(glob.glob(os.path.join(data_shards, "*")))
    result_path = "/data/huongn/data_processing/extra_files.txt"

    with multiprocessing.Pool() as pool:
        res = list(
            tqdm(
                pool.imap_unordered(
                    parallel_check_files,
                    zip(
                        wds_dirs,
                        ds_dirs,
                        repeat(result_path),
                    ),
                ),
                total=len(glob.glob(os.path.join(data_shards, "*"))),
            )
        )
