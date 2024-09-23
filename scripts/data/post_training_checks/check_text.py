import numpy as np
import tarfile
import multiprocessing
from tqdm import tqdm
from itertools import repeat
from typing import Tuple
from fire import Fire
from whisper.tokenizer import get_tokenizer
import open_whisper as ow


def read_file(tar_path: str, file_path: str):
    with tarfile.open(tar_path, "r") as tar:
        text_file = tar.extractfile(file_path)
        return (tar_path + "/" + file_path, text_file.read().decode("utf-8"))


def parallel_read(args):
    return read_file(*args)


def process_file(transcript_tuple: Tuple, n_text_ctx: int):
    tokenizer = get_tokenizer(multilingual=False)
    file_path, transcript_string = transcript_tuple
    reader = ow.utils.TranscriptReader(transcript_string=transcript_string, ext="srt")
    transcript, *_ = reader.read()

    if not transcript:
        text_tokens = [tokenizer.no_speech]

        transcript_text = ""
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
        return (file_path, transcript_text, "OK")
    except ValueError:
        with open("faulty_text_transcript.txt", "a") as f:
            f.write(
                f"{file_path} | {len(text_input)} | {len(text_y)} | {transcript_text} |\n"
            )

        return (file_path, transcript_text, len(text_input), len(text_y))


def parallel_process(args):
    return process_file(*args)


def check_text(tar_path: str, n_text_ctx: int):
    print("Getting file names in tar file")
    with tarfile.open(tar_path, "r") as tar:
        file_paths = [name for name in tar.getnames() if name.endswith(".srt")]

    print("Reading files")
    with multiprocessing.Pool() as pool:
        transcript_tuples = list(
            tqdm(
                pool.imap_unordered(parallel_read, zip(repeat(tar_path), file_paths)),
                total=len(file_paths),
            )
        )

    print(transcript_tuples[:10])

    print("Checking transcript texts")
    with multiprocessing.Pool() as pool:
        faulty_transcripts = list(
            tqdm(
                pool.imap_unordered(
                    parallel_process,
                    zip(transcript_tuples, repeat(n_text_ctx)),
                ),
                total=len(transcript_tuples),
            )
        )

    print(faulty_transcripts[:10])

if __name__ == "__main__":
    Fire(check_text)