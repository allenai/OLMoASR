from whisper.tokenizer import get_tokenizer
import multiprocessing
from tqdm import tqdm
import glob
import os
import json
from open_whisper.utils import TranscriptReader
from typing import Optional
from fire import Fire


def over_ml_ctx_len(sample_dicts):
    tokenizer = get_tokenizer(language="en", multilingual=True)
    new_sample_dicts = []

    for smpl_dict in sample_dicts:
        try:
            reader = TranscriptReader(
                file_path=smpl_dict["transcript"],
                ext=smpl_dict["transcript"].split(".")[-1],
            )
            transcript, *_ = reader.read()
            text = reader.extract_text(transcript=transcript)

            text_tokens = tokenizer.encode(text)
            text_tokens = (
                list(tokenizer.sot_sequence_including_notimestamps) + text_tokens
            )
            text_tokens.append(tokenizer.eot)

            if len(text_tokens) > 448:
                continue
            else:
                new_sample_dicts.append(smpl_dict)
        except Exception as e:
            continue

    return new_sample_dicts


def main(
    samples_dicts_dir: str,
    batches: int,
    output_dir: str,
    start_output_dir_idx: Optional[int] = None,
):
    samples_dicts_files = sorted(glob.glob(samples_dicts_dir + "/*/samples_dicts.jsonl"))
    print(f"{len(samples_dicts_files)=}")
    batch_size = (len(samples_dicts_files) // batches) + 1
    print(f"{batch_size=}")
    batch_idx = int(os.getenv("BEAKER_REPLICA_RANK"))
    print(f"{batch_idx=}")
    samples_dicts_files = samples_dicts_files[
        batch_idx * batch_size : (batch_idx * batch_size) + batch_size
    ]
    print(f"{len(samples_dicts_files)=}")
    if start_output_dir_idx:
        start_output_dir_idx = start_output_dir_idx + (batch_size * batch_idx)

    for i, p in tqdm(enumerate(samples_dicts_files), total=len(samples_dicts_files)):
        with open(p, "r") as f:
            samples_dicts = [json.loads(line.strip())["sample_dicts"] for line in f]

        with multiprocessing.Pool() as pool:
            new_samples_dicts = list(
                tqdm(
                    pool.imap_unordered(over_ml_ctx_len, samples_dicts),
                    total=len(samples_dicts),
                )
            )

        new_samples_dicts = [nsd for nsd in new_samples_dicts if len(nsd) > 0]
        print(f"{len(new_samples_dicts)=}")
        os.makedirs(f"{output_dir}/{(start_output_dir_idx + i):03}", exist_ok=True)
        with open(
            f"{output_dir}/{(start_output_dir_idx + i):03}/samples_dicts.jsonl", "w"
        ) as f:
            for nsd in new_samples_dicts:
                f.write(json.dumps({"sample_dicts": nsd}) + "\n")


if __name__ == "__main__":
    Fire(main)
