from ftlangdetect import detect
from open_whisper.utils import TranscriptReader
import multiprocessing
from tqdm import tqdm
import os
import glob
from itertools import chain


def is_en(file_path: str):
    reader = TranscriptReader(
        file_path=file_path, transcript_string=None, ext=file_path.split(".")[-1]
    )
    t_dict, *_ = reader.read()
    text = reader.extract_text(transcript=t_dict)
    res = detect(text=text, low_memory=False)
    if res["lang"] == "en":
        return True
    else:
        return file_path


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

    list_files = lambda p, ext: glob.glob(p + f"/*/*.{ext}")
    transcript_files = list(
        chain(
            *[
                (
                    list_files(p, "srt")
                    if int(p.split("/")[-1]) < 2449
                    else list_files(p, "vtt")
                )
                for p in shard_paths
            ]
        )
    )
    print(transcript_files[:20])
    
    with multiprocessing.Pool() as pool:
        res = list(
            tqdm(pool.imap_unordered(is_en, transcript_files), total=len(transcript_files))
        )

    with open(f"{log_dir}/non_en_files.txt", "a") as f:
        for r in res:
            if r is not True:
                f.write(r + "\n")
