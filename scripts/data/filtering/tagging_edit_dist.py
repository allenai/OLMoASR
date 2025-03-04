import multiprocessing
from tqdm import tqdm
import os
import glob
import jiwer
import webvtt
import json
import gzip
from itertools import repeat
from open_whisper.utils import TranscriptReader
from whisper.normalizers import EnglishTextNormalizer

# import logging

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
# )

# logger = logging.getLogger(__name__)


def get_man_text(man_content):
    reader = TranscriptReader(
        file_path=None,
        transcript_string=man_content,
        ext="vtt" if man_content.startswith("WEBVTT") else "srt",
    )
    t_dict, *_ = reader.read()
    man_text = reader.extract_text(t_dict)
    return man_text


def get_mach_text(mach_content):
    content = webvtt.from_string(mach_content)
    modified_content = []
    if len(content) > 0:
        if len(content) > 1:
            if content[0].text == content[1].text:
                modified_content.append(content[0])
                start = 2
            else:
                start = 0
        elif len(content) == 1:
            start = 0

        for i in range(start, len(content)):
            caption = content[i]
            if "\n" not in caption.text:
                modified_content.append(caption)
            elif "\n" in caption.text and i == len(content) - 1:
                caption.text = caption.text.split("\n")[-1]
                modified_content.append(caption)

        mach_text = " ".join([caption.text for caption in modified_content])
    else:
        mach_text = ""
    return mach_text


def tag_edit_dist(transcript_dict, normalizer):
    man_text = get_man_text(transcript_dict["content"])
    mach_text = get_mach_text(transcript_dict["mach_content"])

    norm_man_text = normalizer(man_text)
    norm_mach_text = normalizer(mach_text)

    if norm_man_text != "":
        edit_dist = jiwer.wer(norm_man_text, norm_mach_text)
    elif man_text == "":
        if norm_mach_text != "":
            edit_dist = jiwer.wer(norm_mach_text, man_text)
        elif mach_text != "":
            edit_dist = jiwer.wer(mach_text, man_text)
        elif mach_text == "":
            edit_dist = 0.0
    elif man_text != "":
        edit_dist = jiwer.wer(man_text, norm_mach_text)
    transcript_dict["man_text"] = norm_man_text if norm_man_text != "" else man_text
    transcript_dict["mach_text"] = norm_mach_text if norm_mach_text != "" else mach_text
    transcript_dict["edit_dist"] = edit_dist

    return transcript_dict


def process_shard(shard_jsonl, normalizer, output_dir):
    count_0 = 0
    count_1 = 0
    count_gt_1 = 0
    count_lt_1 = 0
    with gzip.open(shard_jsonl, "rt") as f:
        transcript_dicts = [json.loads(line.strip()) for line in f]

    for i, transcript_dict in enumerate(transcript_dicts):
        transcript_dicts[i] = tag_edit_dist(transcript_dict, normalizer)
        if transcript_dict["edit_dist"] == 0.0:
            count_0 += 1
        elif transcript_dict["edit_dist"] == 1.0:
            count_1 += 1
        elif transcript_dict["edit_dist"] > 1.0:
            count_gt_1 += 1
        elif transcript_dict["edit_dist"] < 1.0 and transcript_dict["edit_dist"] > 0.0:
            count_lt_1 += 1

    with gzip.open(f"{output_dir}/{os.path.basename(shard_jsonl)}", "wt") as f:
        for transcript_dict in transcript_dicts:
            f.write(json.dumps(transcript_dict) + "\n")

    avg_0, avg_1, avg_gt_1, avg_lt_1 = (
        count_0 / len(transcript_dicts),
        count_1 / len(transcript_dicts),
        count_gt_1 / len(transcript_dicts),
        count_lt_1 / len(transcript_dicts),
    )

    return avg_0, avg_1, avg_gt_1, avg_lt_1


def parallel_process_shard(args):
    return process_shard(*args)


def main(input_dir, output_dir):
    normalizer = EnglishTextNormalizer()
    shard_files = glob.glob(f"{input_dir}/*.jsonl.gz")

    with multiprocessing.Pool() as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    parallel_process_shard,
                    zip(shard_files, repeat(normalizer), repeat(output_dir)),
                )
            ),
            total=len(shard_files),
        )

    avg_0, avg_1, avg_gt_1, avg_lt_1 = zip(*results)
    avg_0 = sum(avg_0) / len(shard_files)
    avg_1 = sum(avg_1) / len(shard_files)
    avg_gt_1 = sum(avg_gt_1) / len(shard_files)
    avg_lt_1 = sum(avg_lt_1) / len(shard_files)

    with open(f"{output_dir}/edit_dist_stats.log", "w") as f:
        f.write(f"Percentage of transcript files w/ edit dist 0: {avg_0}\n")
        f.write(f"Percentage of transcript files w/ edit dist 1: {avg_1}\n")
        f.write(f"Percentage of transcript files w/ edit dist > 1: {avg_gt_1}\n")
        f.write(
            f"Percentage of transcript files w/ edit dist < 1 and > 0: {avg_lt_1}\n"
        )

    print(f"{avg_0=}, {avg_1=}, {avg_gt_1=}, {avg_lt_1=}")
