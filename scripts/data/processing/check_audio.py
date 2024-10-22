import numpy as np
import multiprocessing
from tqdm import tqdm
from itertools import chain
from fire import Fire
from whisper import audio
import glob
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.FileHandler("main.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def check_audio(audio_file: str):
    try:
        audio_arr = np.load(audio_file).astype(np.float32) / 32768.0
        audio_arr = audio.pad_or_trim(audio_arr)
        mel_spec = audio.log_mel_spectrogram(audio_arr)

        return None
    except Exception as e:
        return audio_file


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

    audio_files = list(chain(*[glob.glob(p + "/*/*.npy") for p in shard_paths]))

    logger.info("Checking audio files")
    with multiprocessing.Pool() as pool:
        faulty_audio = list(
            tqdm(
                pool.imap_unordered(
                    check_audio,
                    audio_files,
                ),
                total=len(audio_files),
            )
        )

    faulty_audio = [a for a in faulty_audio if a is not None]

    if len(faulty_audio) > 0:
        with open(
            f"{log_dir}/faulty_audio_train_{shard_paths[0].split('/')[-1]}-{shard_paths[-1].split('/')[-1]}.txt",
            "w",
        ) as f:
            for a in faulty_audio:
                f.write(f"{a}\n")

        logger.info("Faulty audio found")
    else:
        logger.info("No faulty audio found")


if __name__ == "__main__":
    Fire(main)
