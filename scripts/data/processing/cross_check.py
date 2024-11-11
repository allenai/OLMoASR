import multiprocessing
from tqdm import tqdm
import os
import shutil
import glob
from itertools import repeat
from fire import Fire


def check_files(full_shard_dir, seg_shard_dir, spill_dir, dry_run):
    full_shard_dir_ids = set(os.listdir(full_shard_dir))
    seg_shard_dir_ids = set(os.listdir(seg_shard_dir))
    shard = os.path.basename(full_shard_dir)
    print(f"{shard=}")
    spill_dir = os.path.join(spill_dir, shard)
    print(f"{spill_dir=}")
    os.makedirs(spill_dir, exist_ok=True)

    extra_ids = full_shard_dir_ids - seg_shard_dir_ids
    print(f"{extra_ids=}")

    if len(extra_ids) > 0:
        if not dry_run:
            for video_id in extra_ids:
                shutil.move(os.path.join(full_shard_dir, video_id), spill_dir)
        else:
            print(f"Extra ids in {shard}: {extra_ids}")


def parallel_check_files(args):
    return check_files(*args)


def main(
    full_dir_path: str,
    seg_dir_path: str,
    spill_dir: str,
    batch_size: int,
    start_shard_idx: int,
    end_shard_idx: int,
    dry_run: bool,
):
    os.makedirs(spill_dir, exist_ok=True)
    
    job_idx = int(os.getenv("BEAKER_REPLICA_RANK"))
    full_shard_dirs = sorted(glob.glob(os.path.join(full_dir_path, "*")))[
        start_shard_idx : end_shard_idx + 1
    ][job_idx * batch_size : (job_idx + 1) * batch_size]
    print(f"Processing from {full_shard_dirs[0]} to {full_shard_dirs[-1]}")
    seg_shard_dirs = sorted(glob.glob(os.path.join(seg_dir_path, "*")))[
        start_shard_idx : end_shard_idx + 1
    ][job_idx * batch_size : (job_idx + 1) * batch_size]
    print(f"Processing from {seg_shard_dirs[0]} to {seg_shard_dirs[-1]}")

    with multiprocessing.Pool() as pool:
        res = list(
            tqdm(
                pool.imap_unordered(
                    parallel_check_files,
                    zip(
                        full_shard_dirs,
                        seg_shard_dirs,
                        repeat(spill_dir),
                        repeat(dry_run),
                    ),
                ),
                total=len(full_shard_dirs),
            )
        )


if __name__ == "__main__":
    Fire(main)
