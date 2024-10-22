import ray
from ray.data.datasource import FilenameProvider
import glob
import os
from typing import Dict, Any, Optional
from fire import Fire
from itertools import chain
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.FileHandler("main.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

class FilenameProviders:
    def __init__(self):
        pass

    @staticmethod
    class SamplesDictsFilenameProvider(FilenameProvider):
        def __init__(self, file_format: str):
            self.file_format = file_format

        def get_filename_for_block(self, block, task_index, block_index):
            return f"samples_dicts.{self.file_format}"


def gen_smpl_dict(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    segs_dir = os.path.dirname(row["item"]).replace("440K_full", "440K_seg")

    if int(row["item"].split("/")[-2]) > 2448:
        text_files = sorted(glob.glob(segs_dir + "/*.vtt"))
    else:
        text_files = sorted(glob.glob(segs_dir + "/*.srt"))
    npy_files = sorted(glob.glob(segs_dir + "/*.npy"))
    text_npy_samples = list(zip(text_files, npy_files))
    smpl_dicts = []

    for text_fp, npy_fp in text_npy_samples:
        smpl_dict = {"key": segs_dir, "transcript": text_fp, "audio": npy_fp}
        smpl_dicts.append(smpl_dict)

    row["sample_dicts"] = smpl_dicts
    del row["item"]

    return row


def main(
    shard_metadata: str,
    samples_dicts_dir: str,
    batch_size: int,
):
    batch_idx = int(os.getenv("BEAKER_REPLICA_RANK"))
    with open(shard_metadata, "r") as f:
        shard_metadata = [line.strip() for line in f]
        
    shard_dirs = shard_metadata[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    logger.info(f"{shard_dirs[0]=}, {shard_dirs[-1]=}, {len(shard_dirs)=}")
    data_dirs = list(chain(*[glob.glob(shard_dir + "/*") for shard_dir in shard_dirs]))
    logger.info(f"{data_dirs[:5]=}")
    ray.init()

    logger.info("Generating sample dicts")
    ds = ray.data.from_items(data_dirs).map(gen_smpl_dict)

    logger.info("Writing sample dicts to disk")
    ds.repartition(num_blocks=1).write_json(
        samples_dicts_dir,
        filename_provider=FilenameProviders.SamplesDictsFilenameProvider("jsonl"),
    )

if __name__ == "__main__":
    Fire(main)
