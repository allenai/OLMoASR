import multiprocessing
from tqdm import tqdm
import glob
import os
import json
from fire import Fire


def to_dicts(file_path: str):
    return {
        "path": file_path,
        "text": open(file_path, "r").read(),
        "dur": dur_dict[os.path.dirname(file_path).split("/")[-1]],
    }


def main(shard_dir: str, text_dicts_dir: str):
    if int(os.path.basename(shard_dir)) < 2449:
        ext = "srt"
    else:
        ext = "vtt"
    transcript_files = glob.glob(os.path.join(shard_dir, f"*/*.{ext}"))
    print(transcript_files[:20])
    print()

    with multiprocessing.Pool() as pool:
        text_dicts = list(
            tqdm(
                pool.imap_unordered(to_dicts, transcript_files),
                total=len(transcript_files),
            )
        )
    print(text_dicts[:20])

    with open(f"{text_dicts_dir}/{os.path.basename(shard_dir)}.json", "w") as f:
        for d in text_dicts:
            f.write(json.dumps(d) + "\n")

if __name__ == "__main__":
    with open(os.getenv("DUR_FILE"), "r") as f:
        dur_dict = json.load(f)
     
    Fire(main)