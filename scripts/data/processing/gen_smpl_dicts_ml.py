import numpy as np
import glob
import multiprocessing
from tqdm import tqdm
from itertools import chain
import pycld2 as cld2
from open_whisper.utils import TranscriptReader
import json
from itertools import repeat
from whisper.tokenizer import LANGUAGES
from fire import Fire


def get_all_paths(path: str):
    return glob.glob(path + "/*")


def gen_smpl_dict(segs_dir):
    text_files = sorted(glob.glob(segs_dir + "/*.vtt"))
    npy_files = sorted(glob.glob(segs_dir + "/*.npy"))
    language = segs_dir.split("/")[-2].split("_")[-1]
    text_npy_samples = list(zip(text_files, npy_files))
    smpl_dicts = []

    for text_fp, npy_fp in text_npy_samples:
        smpl_dict = {"transcript": text_fp, "audio": npy_fp, "language": language}
        smpl_dicts.append(smpl_dict)

    return smpl_dicts


def same_lang(smpl_dicts):
    # new_smpl_dicts = []
    if smpl_dicts[0]["language"] not in LANGUAGES:
        return None
    for i, smpl_dict in enumerate(smpl_dicts):
        reader = TranscriptReader(
            file_path=smpl_dict["transcript"],
            ext=smpl_dict["transcript"].split(".")[-1],
        )
        transcript, *_ = reader.read()
        text = reader.extract_text(transcript=transcript)
        if text != "":
            isReliable, _, details = cld2.detect(text)
            # if not isReliable:
            #     print(f"Unreliable detection")
            # else:
            lang_code = details[0][1]
            if lang_code != smpl_dict["language"]:
                return None
                # print(f"Language mismatch: {lang_code} vs {smpl_dict['language']}")
            else:
                if i == len(smpl_dicts) - 1:
                    return smpl_dicts
                else:
                    continue
        else:
            if i == len(smpl_dicts) - 1:
                return smpl_dicts
            else:
                continue


def write_to_jsonl(smpl_dicts, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/samples_dicts.jsonl", "w") as f:
        for smpl_dict in smpl_dicts:
            f.write(json.dumps({"sample_dicts": smpl_dict}) + "\n")


def parallel_write_to_jsonl(args):
    return write_to_jsonl(*args)


def main(src_dir, split_factor, output_dir):
    segs_dir = glob.glob(src_dir + "/*/*")
    print(f"{len(segs_dir)=}")
    print(f"{segs_dir[:5]=}")

    print("Generating sample dictionaries")
    with multiprocessing.Pool() as pool:
        smpl_dicts = list(
            tqdm(pool.imap_unordered(gen_smpl_dict, segs_dir), total=len(segs_dir))
        )

    print(f"{len(smpl_dicts)=}")
    print(f"{smpl_dicts[:5]=}")

    print("Filtering sample dictionaries")
    with multiprocessing.Pool() as pool:
        smpl_dicts = list(
            tqdm(pool.imap_unordered(same_lang, smpl_dicts), total=len(smpl_dicts))
        )

    print(f"{len(smpl_dicts)=}")
    print(f"{smpl_dicts[:5]=}")

    print("Writing sample dictionaries to jsonl")
    smpl_dicts = list(filter(None, smpl_dicts))

    batch_size = len(smpl_dicts) // split_factor
    smpl_dicts_batches = [
        (smpl_dicts[i : i + batch_size], i // batch_size)
        for i in range(0, len(smpl_dicts), batch_size)
    ]

    print(f"{len(smpl_dicts_batches)=}")
    print(f"{smpl_dicts_batches[:5]=}")

    with multiprocessing.Pool() as pool:
        res = list(
            tqdm(
                pool.imap_unordered(
                    parallel_write_to_jsonl,
                    smpl_dicts_batches,
                ),
                total=len(smpl_dicts_batches),
            )
        )


if __name__ == "__main__":
    Fire(main)
