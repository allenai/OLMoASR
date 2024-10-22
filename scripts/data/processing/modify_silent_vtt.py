import multiprocessing
from tqdm import tqdm
import glob
from itertools import chain
import logging
from fire import Fire
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.FileHandler("main.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def modify_silent_vtt(vtt_file):
    with open(vtt_file, "r") as f:
        lines = [line.strip() for line in f]
    if len(lines) == 0:
        # with open(vtt_file, "w") as f:
        #     f.write("WEBVTT\n\n")
        return vtt_file


def main(
    source_dir: str,
    log_dir: str,
    start_shard_idx: int,
    end_shard_idx: int,
    batch_size: int,
):
    job_idx = int(os.getenv("BEAKER_REPLICA_RANK"))
    os.makedirs(log_dir, exist_ok=True)
    shard_paths = sorted(glob.glob(source_dir + "/*"))[
        start_shard_idx : end_shard_idx + 1
    ][job_idx * batch_size : (job_idx + 1) * batch_size]
    logger.info(f"{shard_paths[0]=}, {shard_paths[-1]=}, {len(shard_paths)=}")
    vtt_files = list(chain(*[glob.glob(p + "/*/*.vtt") for p in shard_paths]))

    with multiprocessing.Pool() as pool:
        res = list(
            tqdm(
                pool.imap_unordered(modify_silent_vtt, vtt_files), total=len(vtt_files)
            )
        )

    silent_vtt = [r for r in res if r is not None]

    logger.info(f"{len(silent_vtt)} VTT files were empty")
    logger.info(
        f"{((len(silent_vtt) / len(vtt_files)) * 100):.2f}% of VTT files were empty"
    )

    with open(
        f"{log_dir}/silent_vtt_files_{shard_paths[0].split('/')[-1]}-{shard_paths[-1].split('/')[-1]}.txt",
        "w",
    ) as f:
        for vtt in silent_vtt:
            f.write(f"{vtt}\n")
            
if __name__ == "__main__":
    Fire(main)
