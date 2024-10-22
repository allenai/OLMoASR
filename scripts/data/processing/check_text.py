import numpy as np
import multiprocessing
from tqdm import tqdm
from itertools import repeat, chain
from fire import Fire
from whisper.tokenizer import get_tokenizer
import open_whisper as ow
import glob
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.FileHandler("main.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def check_text(transcript_file: str, n_text_ctx: int, ext: str):
    tokenizer = get_tokenizer(multilingual=False)
    reader = ow.utils.TranscriptReader(
        file_path=transcript_file, transcript_string=None, ext=ext
    )
    transcript, *_ = reader.read()

    if len(transcript.keys()) == 0:
        return None
    else:
        transcript_text = reader.extract_text(transcript=transcript)

        text_tokens = tokenizer.encode(transcript_text)

    text_tokens = list(tokenizer.sot_sequence_including_notimestamps) + text_tokens

    text_tokens.append(tokenizer.eot)

    text_input = text_tokens[:-1]
    text_y = text_tokens[1:]

    try:
        text_input = np.pad(
            text_input,
            pad_width=(0, n_text_ctx - len(text_input)),
            mode="constant",
            constant_values=51864,
        )
        text_y = np.pad(
            text_y,
            pad_width=(0, n_text_ctx - len(text_y)),
            mode="constant",
            constant_values=51864,
        )
        return None
    except ValueError:
        return (transcript_file, len(text_input), len(text_y))


def parallel_check(args):
    return check_text(*args)


def main(
    source_dir: str,
    log_dir: str,
    start_shard_idx: int,
    end_shard_idx: int,
    batch_size: int,
    n_text_ctx: int,
    ext: str,
):
    job_idx = int(os.getenv("BEAKER_REPLICA_RANK"))
    os.makedirs(log_dir, exist_ok=True)
    shard_paths = sorted(glob.glob(source_dir + "/*"))[
        start_shard_idx : end_shard_idx + 1
    ][job_idx * batch_size : (job_idx + 1) * batch_size]
    logger.info(f"{shard_paths[0]=}, {shard_paths[-1]=}, {len(shard_paths)=}")

    transcript_files = list(chain(*[glob.glob(p + "/*/*.vtt") for p in shard_paths]))
    
    # to check because it seems YT switched to VTT format for all subtitles? (post 440K download)
    srt_files = list(chain(*[glob.glob(p + "/*/*.srt") for p in shard_paths]))
    
    if len(srt_files) > 0:
        logger.info("SRT files found")
        with open(
            f"{log_dir}/srt_files_{shard_paths[0].split('/')[-1]}-{shard_paths[-1].split('/')[-1]}.txt",
            "w",
        ) as f:
            for srt in srt_files:
                f.write(f"{srt}\n")

    logger.info("Checking transcript texts")
    with multiprocessing.Pool() as pool:
        faulty_transcripts = list(
            tqdm(
                pool.imap_unordered(
                    parallel_check,
                    zip(transcript_files, repeat(n_text_ctx), repeat(ext)),
                ),
                total=len(transcript_files),
            )
        )

    faulty_transcripts = [t for t in faulty_transcripts if t is not None]

    if len(faulty_transcripts) > 0:
        with open(
            f"{log_dir}/faulty_transcripts_train_{shard_paths[0].split('/')[-1]}-{shard_paths[-1].split('/')[-1]}.txt",
            "w",
        ) as f:
            for t in faulty_transcripts:
                f.write(f"{t}\n")

        logger.info("Faulty transcripts found")
    else:
        logger.info("No faulty transcripts found")


if __name__ == "__main__":
    Fire(main)
