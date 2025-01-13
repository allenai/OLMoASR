from get_eval_train import get_eval_train
import os
import numpy as np
import glob
import json
from open_whisper.utils import TranscriptReader
from torchaudio.datasets import TEDLIUM
from fire import Fire
import re
import string
import multiprocessing
from tqdm import tqdm
from typing import Optional
from itertools import repeat, chain

# preprocess both positive and negative training data to ensure they are in the same format (no transcript specific format remains)
# generate text file w/ labels from these 2 sets of data


def gen_text(
    transcript_file: Optional[str] = None, transcript_string: Optional[str] = None
):
    reader = TranscriptReader(
        file_path=transcript_file if transcript_file else None,
        transcript_string=None if transcript_file else transcript_string,
        ext=transcript_file.split(".")[-1] if transcript_file else None,
    )
    t_dict, *_ = reader.read()
    text = reader.extract_text(t_dict)
    text = text.lower()
    punctuation_to_remove = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans("", "", punctuation_to_remove))
    return text


def parallel_gen_text(args):
    return gen_text(*args)


def main(
    eval_set: str,
    eval_train_dir: str,
    train_dir: str,
    segment_filter: bool,
    jsonl_input: bool,
):
    # collect all positive training data (data from eval set)
    if eval_set == "tedlium":
        if not os.path.exists(f"{eval_train_dir}/TEDLIUM_release-3"):
            get_eval_train(eval_set=eval_set, eval_dir=eval_train_dir)

        # Initialize the dataset
        dataset = TEDLIUM(root=f"{eval_train_dir}", release="release3", subset="train")

        # Specify the output text file
        output_file = f"{eval_train_dir}/tedlium_train.txt"

        # Open the file for writing
        with open(output_file, "w", encoding="utf-8") as file:
            for index in range(len(dataset)):
                # Get the data for the current index
                _, _, text_y, *_ = dataset[index]

                text_y = re.sub(r"<unk>\s*", "", text_y)
                text_y = text_y.strip()
                text_y += "\n"
                # Write the transcript to the file
                file.write("__label__positive " + text_y)

        print(f"Transcripts have been written to {output_file}.")

        # get count of documents in eval train data
        negative_subsample_count = len(dataset)
        print(f"Number of documents in eval train data: {negative_subsample_count}")

    # subsample negative training data (from training pool) and match num. of docs w/ positive training data (from samples dict)
    if segment_filter:
        print("Segment filtering is enabled.")
        samples_dicts_dirs = glob.glob(f"{train_dir}/*")
        sample_dicts_dir = np.random.choice(samples_dicts_dirs, 1)[0]
        print(f"{sample_dicts_dir=}")
        with open(f"{sample_dicts_dir}/samples_dicts.jsonl", "r") as f:
            sample_dicts = list(
                chain(*[json.loads(line.strip())["sample_dicts"] for line in f])
            )
        print(f"{len(sample_dicts)=}")
        print(f"{sample_dicts[:5]=}")

        subsampled_train_data = np.random.choice(
            sample_dicts, negative_subsample_count, replace=False
        )
        print(f"{len(subsampled_train_data)=}")
        print(f"{subsampled_train_data[:5]=}")
        subsampled_train_data = [
            sample_dict["transcript"] for sample_dict in subsampled_train_data
        ]
        print(f"{len(subsampled_train_data)=}")
        print(f"{subsampled_train_data[:5]=}")

        with multiprocessing.Pool() as pool:
            subsampled_train_text = list(
                tqdm(
                    pool.imap_unordered(
                        parallel_gen_text, zip(subsampled_train_data, repeat(None))
                    ),
                    total=len(subsampled_train_data),
                )
            )
        print(f"{len(subsampled_train_text)=}")
        print(f"{subsampled_train_text[:5]=}")
    else:
        if jsonl_input:
            shard_jsonls = glob.glob(f"{train_dir}/*")
            subsampled_train_data = []
            subsampled_count = 0
            while True:
                shard_jsonl = np.random.choice(shard_jsonls, 1)[0]
                with open(shard_jsonl, "r") as f:
                    transcript_strings = [
                        (
                            json.loads(line.strip())["subtitle_file"],
                            json.loads(line.strip())["content"],
                        )
                        for line in f
                    ]
                if len(transcript_strings) < (
                    negative_subsample_count - subsampled_count
                ):
                    subsampled_train_data.extend(transcript_strings)
                    subsampled_count += len(transcript_strings)
                elif subsampled_count < negative_subsample_count:
                    subsampled_train_data.extend(
                        np.random.choice(
                            transcript_strings,
                            negative_subsample_count - subsampled_count,
                            replace=False,
                        )
                    )
                    break
                else:
                    subsampled_train_data.extend(
                        np.random.choice(
                            transcript_strings, negative_subsample_count, replace=False
                        )
                    )
                    break

            with multiprocessing.Pool() as pool:
                subsampled_train_text = list(
                    tqdm(
                        pool.imap_unordered(
                            parallel_gen_text,
                            subsampled_train_data,
                        ),
                        total=len(subsampled_train_data),
                    )
                )
        else:
            shard_dirs = glob.glob(f"{train_dir}/*")
            subsampled_train_data = []
            subsampled_count = 0
            while True:
                shard_dir = np.random.choice(shard_dirs, 1)[0]
                ext = "vtt" if int(shard_dir.split("/")[-1]) > 2448 else "srt"
                transcript_files = glob.glob(f"{shard_dir}/*/*.{ext}")
                if len(transcript_files) < (
                    negative_subsample_count - subsampled_count
                ):
                    subsampled_train_data.extend(transcript_files)
                    subsampled_count += len(transcript_files)
                elif subsampled_count < negative_subsample_count:
                    subsampled_train_data.extend(
                        np.random.choice(
                            transcript_files,
                            negative_subsample_count - subsampled_count,
                            replace=False,
                        )
                    )
                    break
                else:
                    subsampled_train_data.extend(
                        np.random.choice(
                            transcript_files, negative_subsample_count, replace=False
                        )
                    )
                    break

            with multiprocessing.Pool() as pool:
                subsampled_train_text = list(
                    tqdm(
                        pool.imap_unordered(
                            parallel_gen_text, zip(subsampled_train_data, repeat(None))
                        ),
                        total=len(subsampled_train_data),
                    )
                )

    print("Generating text from subsampled training data... (negative examples)")
    # generate text file w/ labels from negative training data
    with open(output_file, "a") as file:
        for text in subsampled_train_text:
            file.write("__label__negative " + text + "\n")


if __name__ == "__main__":
    Fire(main)
