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
from fire import Fire
import pycld2 as cld2
from typing import Literal

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


def tag_text_lang(transcript_dict):
    man_text = get_man_text(transcript_dict["content"])

    *_, details = cld2.detect(man_text)
    lang_id = details[0][1]

    transcript_dict["text_lang"] = lang_id

    if lang_id == "en":
        en_count = 1
    else:
        en_count = 0
    non_en_count = 1 - en_count

    return transcript_dict, en_count, non_en_count


def process_shard(
    shard_jsonl, normalizer, output_dir, tagger: Literal["edit_dist", "text_lang"]
):
    stats = None

    with gzip.open(shard_jsonl, "rt") as f:
        transcript_dicts = [json.loads(line.strip()) for line in f]

    if tagger == "edit_dist":
        count_0 = 0
        count_1 = 0
        count_gt_1 = 0
        count_lt_1 = 0

        for i, transcript_dict in enumerate(transcript_dicts):
            transcript_dicts[i] = tag_edit_dist(transcript_dict, normalizer)
            if transcript_dict["edit_dist"] == 0.0:
                count_0 += 1
            elif transcript_dict["edit_dist"] == 1.0:
                count_1 += 1
            elif transcript_dict["edit_dist"] > 1.0:
                count_gt_1 += 1
            elif (
                transcript_dict["edit_dist"] < 1.0
                and transcript_dict["edit_dist"] > 0.0
            ):
                count_lt_1 += 1

        avg_0, avg_1, avg_gt_1, avg_lt_1 = (
            count_0 / len(transcript_dicts),
            count_1 / len(transcript_dicts),
            count_gt_1 / len(transcript_dicts),
            count_lt_1 / len(transcript_dicts),
        )

        stats = (avg_0, avg_1, avg_gt_1, avg_lt_1, len(transcript_dicts))
    elif tagger == "text_lang":
        en_counts = 0
        non_en_counts = 0
        for i, transcript_dict in enumerate(transcript_dicts):
            transcript_dicts[i], en_count, non_en_count = tag_text_lang(transcript_dict)
            en_counts += en_count
            non_en_counts += non_en_count
        stats = (
            en_counts / len(transcript_dicts),
            non_en_counts / len(transcript_dicts),
            len(transcript_dicts),
        )

    with gzip.open(f"{output_dir}/{os.path.basename(shard_jsonl)}", "wt") as f:
        for transcript_dict in transcript_dicts:
            f.write(json.dumps(transcript_dict) + "\n")

    return stats


def parallel_process_shard(args):
    return process_shard(*args)


def main(input_dir, output_dir, tagger: Literal["edit_dist", "text_lang"]):
    normalizer = EnglishTextNormalizer()
    shard_files = glob.glob(f"{input_dir}/*.jsonl.gz")
    os.makedirs(output_dir, exist_ok=True)

    with multiprocessing.Pool() as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    parallel_process_shard,
                    zip(
                        shard_files,
                        repeat(normalizer),
                        repeat(output_dir),
                        repeat(tagger),
                    ),
                ),
                total=len(shard_files),
            ),
        )

    if tagger == "edit_dist":
        avg_0, avg_1, avg_gt_1, avg_lt_1, video_id_counts = zip(*results)
        avg_0 = sum(avg_0) / len(shard_files)
        avg_1 = sum(avg_1) / len(shard_files)
        avg_gt_1 = sum(avg_gt_1) / len(shard_files)
        avg_lt_1 = sum(avg_lt_1) / len(shard_files)
        video_id_count = sum(video_id_counts)

        with open(f"{output_dir}/taggger_{tagger}_stats.log", "w") as f:
            f.write(f"Percentage of transcript files w/ edit dist 0: {avg_0}\n")
            f.write(f"Percentage of transcript files w/ edit dist 1: {avg_1}\n")
            f.write(f"Percentage of transcript files w/ edit dist > 1: {avg_gt_1}\n")
            f.write(
                f"Percentage of transcript files w/ edit dist < 1 and > 0: {avg_lt_1}\n"
            )
            f.write(f"Total number of transcript files: {video_id_count}\n")

        print(f"{avg_0=}, {avg_1=}, {avg_gt_1=}, {avg_lt_1=}, {video_id_count=}")
    elif tagger == "text_lang":
        avg_en, avg_non_en, video_id_count = zip(*results)
        avg_en = sum(avg_en) / len(shard_files)
        avg_non_en = sum(avg_non_en) / len(shard_files)
        video_id_count = sum(video_id_count)

        with open(f"{output_dir}/tagger_{tagger}_stats.log", "w") as f:
            f.write(f"Percentage of transcript files w/ English text: {avg_en}\n")
            f.write(
                f"Percentage of transcript files w/ non-English text: {avg_non_en}\n"
            )
            f.write(f"Total number of transcript files: {video_id_count}\n")

        print(f"{avg_en=}, {avg_non_en=}, {video_id_count=}")


if __name__ == "__main__":
    Fire(main)
