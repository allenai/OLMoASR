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
from datasets import load_dataset
import gzip

# preprocess both positive and negative training data to ensure they are in the same format (no transcript specific format remains)
# generate text file w/ labels from these 2 sets of data


class Librispeech:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load(self):
        transcript_files = []
        audio_text = {}
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".txt"):
                    transcript_files.append(os.path.join(root, file))

        for file in sorted(transcript_files):
            with open(file, "r") as f:
                for line in f:
                    audio_codes = line.split(" ")[0].split("-")
                    audio_file = os.path.join(
                        self.root_dir,
                        audio_codes[0],
                        audio_codes[1],
                        f"{audio_codes[0]}-{audio_codes[1]}-{audio_codes[2]}.flac",
                    )
                    audio_text[audio_file] = " ".join(line.split(" ")[1:]).strip()

        return list(audio_text.keys()), list(audio_text.values())


def modify_text(content):
    pattern_brackets = (
        r"[ ]*\[(?![Mm][Uu][Ss][Ii][Cc]\])([A-Z][a-zA-Z]*(?: [A-Z][a-zA-Z]*)*)\][ ]*"
    )
    pattern_parentheses = r"[ ]*\(.*?\)[ ]*"
    pattern_colon = r"[ ]*(?:[A-Z][a-zA-Z]*[ ])+:[ ]*"
    specific_strings = r"[ ]*(?:&nbsp;|&amp;|&lt;|&gt;|=|\.{3}|\\h)+[ ]*"
    primary_pattern = (
        f"{pattern_brackets}|{pattern_parentheses}|{pattern_colon}|{specific_strings}"
    )
    brackets_pattern_capture = r"\[([a-z]+(?: [a-z]+)*)\]"

    content = re.sub(primary_pattern, " ", content)
    content = re.sub(brackets_pattern_capture, r"\1", content)

    return content


def gen_text(
    transcript_file: Optional[str] = None,
    transcript_string: Optional[str] = None,
    max_char_len: Optional[int] = None,
):
    ext = lambda text: "vtt" if text.startswith("WEBVTT") else "srt"
    reader = TranscriptReader(
        file_path=transcript_file if transcript_file else None,
        transcript_string=None if transcript_file else transcript_string,
        ext=(
            transcript_file.split(".")[-1]
            if transcript_file
            else ext(transcript_string)
        ),
    )
    t_dict, *_ = reader.read()
    text = reader.extract_text(t_dict)
    text = modify_text(text)
    text = text.strip()
    text = text.lower()
    punctuation_to_remove = string.punctuation.replace("'", "") + "“" + "”"
    text = text.translate(str.maketrans("", "", punctuation_to_remove))
    text = re.sub(r"\s*\n\s*", " ", text)
    text = text[:max_char_len] if max_char_len else text
    return text


def parallel_gen_text(args):
    return gen_text(*args)


def main(
    eval_set: str,
    eval_train_dir: str,
    train_dir: str,
    segment_filter: bool,
    jsonl_gz_input: bool,
    hf_token: Optional[str] = None,
):
    # collect all positive training data (data from eval set)
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")

    max_char_len = 0

    if eval_set == "tedlium":
        if not os.path.exists(f"{eval_train_dir}/TEDLIUM_release-3"):
            get_eval_train(eval_set=eval_set, eval_dir=eval_train_dir)

        # Initialize the dataset
        dataset = TEDLIUM(root=eval_train_dir, release="release3", subset="train")

        # Specify the output text file
        output_file = f"{eval_train_dir}/tedlium_train.txt"

        # Open the file for writing
        with open(output_file, "w", encoding="utf-8") as file:
            for index in range(len(dataset)):
                # Get the data for the current index
                _, _, text_y, *_ = dataset[index]

                text_y = re.sub(r"<unk>\s*", "", text_y)
                text_y = text_y.replace(" '", "'")
                text_y = text_y.strip()
                text_y += "\n"
                # Write the transcript to the file
                file.write("__label__positive " + text_y)
                if len(text_y) > max_char_len:
                    max_char_len = len(text_y)

        print(f"Transcripts have been written to {output_file}.")
        print(f"{max_char_len=}")

        # get count of documents in eval train data
        negative_subsample_count = len(dataset)
        print(f"Number of documents in eval train data: {negative_subsample_count}")
        random_seed = 28
    elif eval_set == "common_voice":
        if not os.path.exists(
            f"{eval_train_dir}/mozilla-foundation___common_voice_5_1"
        ):
            get_eval_train(
                eval_set=eval_set, eval_dir=eval_train_dir, hf_token=hf_token
            )

        dataset = load_dataset(
            path="mozilla-foundation/common_voice_5_1",
            name="en",
            split="train",
            token=hf_token,
            cache_dir=eval_train_dir,
            trust_remote_code=True,
            num_proc=15,
            save_infos=True,
        )

        output_file = f"{eval_train_dir}/common_voice_train.txt"
        punctuation_to_remove = string.punctuation.replace("'", "") + "“" + "”"

        with open(output_file, "w", encoding="utf-8") as file:
            for index in range(len(dataset)):
                text_y = dataset[index]["sentence"]
                text_y = text_y.strip()
                text_y = text_y.lower()
                text_y = text_y.translate(str.maketrans("", "", punctuation_to_remove))
                text_y += "\n"
                file.write("__label__positive " + text_y)
                if len(text_y) > max_char_len:
                    max_char_len = len(text_y)

        print(f"Transcripts have been written to {output_file}.")
        print(f"{max_char_len=}")

        negative_subsample_count = len(dataset)
        print(f"Number of documents in eval train data: {negative_subsample_count}")
        random_seed = 82
    elif eval_set == "librispeech_clean":
        if not os.path.exists(f"{eval_train_dir}/librispeech_train_clean"):
            get_eval_train(eval_set=eval_set, eval_dir=eval_train_dir)

        _, transcripts = Librispeech(f"{eval_train_dir}/librispeech_train_clean").load()
        output_file = f"{eval_train_dir}/librispeech_clean_train.txt"
        punctuation_to_remove = string.punctuation.replace("'", "") + "“" + "”"

        with open(output_file, "w", encoding="utf-8") as file:
            for transcript in transcripts:
                text_y = transcript.strip()
                text_y = text_y.lower()
                text_y = text_y.translate(str.maketrans("", "", punctuation_to_remove))
                text_y += "\n"
                file.write("__label__positive " + text_y)
                if len(text_y) > max_char_len:
                    max_char_len = len(text_y)

        print(f"Transcripts have been written to {output_file}.")
        print(f"{max_char_len=}")

        negative_subsample_count = len(transcripts)
        print(f"Number of documents in eval train data: {negative_subsample_count}")
        random_seed = 2002
    elif eval_set == "librispeech_other":
        if not os.path.exists(f"{eval_train_dir}/librispeech_train_other"):
            get_eval_train(eval_set=eval_set, eval_dir=eval_train_dir)

        _, transcripts = Librispeech(f"{eval_train_dir}/librispeech_train_other").load()
        output_file = f"{eval_train_dir}/librispeech_other_train.txt"
        punctuation_to_remove = string.punctuation.replace("'", "") + "“" + "”"

        with open(output_file, "w", encoding="utf-8") as file:
            for transcript in transcripts:
                text_y = transcript.strip()
                text_y = text_y.lower()
                text_y = text_y.translate(str.maketrans("", "", punctuation_to_remove))
                text_y += "\n"
                file.write("__label__positive " + text_y)
                if len(text_y) > max_char_len:
                    max_char_len = len(text_y)

        print(f"Transcripts have been written to {output_file}.")
        print(f"{max_char_len=}")

        negative_subsample_count = len(transcripts)
        print(f"Number of documents in eval train data: {negative_subsample_count}")
        random_seed = 2828

    # subsample negative training data (from training pool) and match num. of docs w/ positive training data (from samples dict)
    rng = np.random.default_rng(random_seed)
    if segment_filter:
        print("Segment filtering is enabled.")
        samples_dicts_dirs = glob.glob(f"{train_dir}/*")
        sample_dicts_dir = rng.choice(samples_dicts_dirs, 1)[0]
        print(f"{sample_dicts_dir=}")
        with open(f"{sample_dicts_dir}/samples_dicts.jsonl", "r") as f:
            sample_dicts = list(
                chain(*[json.loads(line.strip())["sample_dicts"] for line in f])
            )
        print(f"{len(sample_dicts)=}")
        print(f"{sample_dicts[:5]=}")

        subsampled_train_data = rng.choice(
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
                        parallel_gen_text,
                        zip(subsampled_train_data, repeat(None), repeat(max_char_len)),
                    ),
                    total=len(subsampled_train_data),
                )
            )
        print(f"{len(subsampled_train_text)=}")
        print(f"{subsampled_train_text[:5]=}")
    else:
        if jsonl_gz_input:
            shard_jsonls = glob.glob(f"{train_dir}/*")
            subsampled_train_data = []
            subsampled_count = 0
            while True:
                shard_jsonl = rng.choice(shard_jsonls, 1)[0]
                with gzip.open(shard_jsonl, "rt") as f:
                    transcript_strings = [
                        json.loads(line.strip())["seg_content"] for line in f
                    ]
                if len(transcript_strings) < (
                    negative_subsample_count - subsampled_count
                ):
                    subsampled_train_data.extend(transcript_strings)
                    subsampled_count += len(transcript_strings)
                elif subsampled_count < negative_subsample_count:
                    subsampled_train_data.extend(
                        rng.choice(
                            transcript_strings,
                            negative_subsample_count - subsampled_count,
                            replace=False,
                        )
                    )
                    break
                else:
                    subsampled_train_data.extend(
                        rng.choice(
                            transcript_strings, negative_subsample_count, replace=False
                        )
                    )
                    break

            with multiprocessing.Pool() as pool:
                subsampled_train_text = list(
                    tqdm(
                        pool.imap_unordered(
                            parallel_gen_text,
                            zip(
                                repeat(None),
                                subsampled_train_data,
                                repeat(max_char_len),
                            ),
                        ),
                        total=len(subsampled_train_data),
                    )
                )
        else:
            shard_dirs = glob.glob(f"{train_dir}/*")
            subsampled_train_data = []
            subsampled_count = 0
            while True:
                shard_dir = rng.choice(shard_dirs, 1)[0]
                ext = "vtt" if int(shard_dir.split("/")[-1]) > 2448 else "srt"
                transcript_files = glob.glob(f"{shard_dir}/*/*.{ext}")
                if len(transcript_files) < (
                    negative_subsample_count - subsampled_count
                ):
                    subsampled_train_data.extend(transcript_files)
                    subsampled_count += len(transcript_files)
                elif subsampled_count < negative_subsample_count:
                    subsampled_train_data.extend(
                        rng.choice(
                            transcript_files,
                            negative_subsample_count - subsampled_count,
                            replace=False,
                        )
                    )
                    break
                else:
                    subsampled_train_data.extend(
                        rng.choice(
                            transcript_files, negative_subsample_count, replace=False
                        )
                    )
                    break

            with multiprocessing.Pool() as pool:
                subsampled_train_text = list(
                    tqdm(
                        pool.imap_unordered(
                            parallel_gen_text,
                            zip(
                                subsampled_train_data,
                                repeat(None),
                                repeat(max_char_len),
                            ),
                        ),
                        total=len(subsampled_train_data),
                    )
                )

    # shuffling data
    with open(output_file, "r") as f:
        positive_examples = [line for line in f]
    print(f"{positive_examples[:5]=}")
    negative_examples = [
        f"__label__negative {text}\n" for text in subsampled_train_text
    ]
    print(f"{negative_examples[:5]=}")
    all_data = positive_examples + negative_examples
    print(f"{len(all_data)=}")
    print(f"{all_data[:5]=}")
    rng.shuffle(all_data)

    print("Generating text from subsampled training data... (negative examples)")
    # generate text file w/ labels from negative training data
    with open(output_file, "w") as file:
        for text in all_data:
            file.write(text)

    print("Splitting data into training and testing")
    train_split = 0.8
    split_index = int(len(all_data) * train_split)
    print(f"{split_index=}")
    train_data = all_data[:split_index]
    test_data = all_data[split_index:]

    with open(f"{eval_train_dir}/{eval_set}.train", "w") as file:
        for text in train_data:
            file.write(text)

    with open(f"{eval_train_dir}/{eval_set}.test", "w") as file:
        for text in test_data:
            file.write(text)


if __name__ == "__main__":
    Fire(main)
