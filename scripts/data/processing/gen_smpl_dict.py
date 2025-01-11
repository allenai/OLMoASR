import glob
import os
from typing import Dict, Any, Optional, List
from fire import Fire
from itertools import chain, repeat
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
    # smpl_dicts = []

    # for text_fp, npy_fp in text_npy_samples:
    #     smpl_dict = {"transcript": text_fp, "audio": npy_fp}
    #     smpl_dicts.append(smpl_dict)
    smpl_dicts = [
        {"transcript": text_fp, "audio": npy_fp} for text_fp, npy_fp in text_npy_samples
    ]

    return smpl_dicts


def gen_smpl_dict_jsonl(transcript_segs_dir, audio_seg_dir) -> List:
    audio_seg_dir = (
        audio_seg_dir + f"/{os.path.dirname(transcript_segs_dir).split('/')[-1]}"
    )
    text_files = sorted(glob.glob(transcript_segs_dir + "/*/*"))
    npy_files = [
        p.replace(transcript_segs_dir, audio_seg_dir).split(".")[0] + ".npy"
        for p in text_files
    ]
    text_npy_samples = list(zip(text_files, npy_files))
    smpl_dicts = [
        {"transcript": text_fp, "audio": npy_fp} for text_fp, npy_fp in text_npy_samples
    ]

    return smpl_dicts


def parallel_gen_smpl_dict_jsonl(args):
    return gen_smpl_dict_jsonl(*args)


def main(
    samples_dicts_dir: str,
    batch_size: int,
    shard_metadata: Optional[str] = None,
    jsonl_seg_dir: Optional[str] = None,
    audio_seg_dir: Optional[str] = None,
):
    batch_idx = int(os.getenv("BEAKER_REPLICA_RANK"))
    samples_dicts_dir = samples_dicts_dir + f"/{batch_idx:03}"
    os.makedirs(samples_dicts_dir, exist_ok=True)
    logger.info(f"{samples_dicts_dir=}")

    if shard_metadata:
        with open(shard_metadata, "r") as f:
            shard_metadata = [line.strip() for line in f]

        shard_dirs = shard_metadata[
            batch_idx * batch_size : (batch_idx + 1) * batch_size
        ]
        logger.info(f"{shard_dirs[0]=}, {shard_dirs[-1]=}, {len(shard_dirs)=}")
        data_dirs = list(
            chain(*[glob.glob(shard_dir + "/*") for shard_dir in shard_dirs])
        )
        logger.info(f"{len(data_dirs)=}")
        logger.info(f"{data_dirs[:5]=}")
        logger.info(f"{data_dirs[-5:]}")

        with multiprocessing.Pool() as pool:
            smpl_dicts = list(
                tqdm(
                    pool.imap_unordered(gen_smpl_dict, data_dirs), total=len(data_dirs)
                )
            )

    else:
        data_dirs = sorted(glob.glob(jsonl_seg_dir + "/*"))[
            batch_idx * batch_size : (batch_idx + 1) * batch_size
        ]
        logger.info(f"{len(data_dirs)=}")
        logger.info(f"{data_dirs[:5]=}")
        logger.info(f"{data_dirs[-5:]}")

        with multiprocessing.Pool() as pool:
            smpl_dicts = list(
                tqdm(
                    pool.imap_unordered(
                        parallel_gen_smpl_dict_jsonl,
                        zip(data_dirs, repeat(audio_seg_dir)),
                    ),
                    total=len(data_dirs),
                )
            )

    logger.info("Writing sample dicts to disk")
    with open(f"{samples_dicts_dir}/samples_dicts.jsonl", "w") as f:
        for smpl_dict in smpl_dicts:
            f.write(json.dumps({"sample_dicts": smpl_dict}) + "\n")


if __name__ == "__main__":
    Fire(main)
