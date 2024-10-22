import multiprocessing
from fire import Fire
import glob
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.FileHandler("main.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def check_segs(audio_files, transcript_files):
    if len(audio_files) != len(transcript_files):
        return os.path.dirname(audio_files[0])
    else:
        return None


def parallel_check(args):
    return check_segs(*args)


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

    audio_files = list([glob.glob(p + "/*/*.npy") for p in shard_paths])
    transcript_files = list([glob.glob(p + "/*/*.vtt") for p in shard_paths])

    logger.info("Checking...")
    with multiprocessing.Pool() as pool:
        uneven_segs = list(
            pool.imap_unordered(
                parallel_check,
                zip(audio_files, transcript_files),
            )
        )

    uneven_segs = [d for d in uneven_segs if d is not None]

    if len(uneven_segs) > 0:
        with open(
            f"{log_dir}/uneven_segs_{shard_paths[0].split('/')[-1]}-{shard_paths[-1].split('/')[-1]}.txt",
            "w",
        ) as f:
            for d in uneven_segs:
                f.write(f"{d}\n")

        logger.info("Uneven segments found")
    else:
        logger.info("No uneven segments found")


if __name__ == "__main__":
    Fire(main)
