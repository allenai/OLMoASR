from multiprocessing import Pool
from functools import partial
import yaml
from tqdm import tqdm
import os
import glob
import re
import jiwer
import webvtt
import pysrt
import json
import gzip
from itertools import repeat
from collections import defaultdict
from fire import Fire
import pycld2 as cld2
from typing import Literal, Optional, Dict, List, Union
from io import StringIO
from open_whisper.utils import TranscriptReader
from whisper.normalizers import EnglishTextNormalizer

# import logging

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
# )

# logger = logging.getLogger(__name__)
CONST_a2z = set([chr(ord("a") + i) for i in range(26)])
CONST_A2Z = set(_.upper() for _ in CONST_a2z)


def run_imap_multiprocessing(func, argument_list, num_processes):
    # Stolen from https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
    pool = Pool(processes=num_processes)

    result_list_tqdm = []
    for result in tqdm(
        pool.imap(func=func, iterable=argument_list), total=len(argument_list)
    ):
        result_list_tqdm.append(result)

    return result_list_tqdm


def parse_config(config_path):
    config_dict = yaml.safe_load(open(config_path, "r"))
    return config_dict


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


def parse_into_iter(content, subtitle_file_name):
    """Parses either the contents of an srt or vtt into an iterable string of things with a .text field"""
    ext = os.path.splitext(subtitle_file_name)[-1]
    if ext == ".srt":
        return pysrt.from_string(content)
    elif ext == ".vtt":
        return webvtt.from_string(content)
    else:
        raise Exception("Unsupported subtitle file type: %s" % subtitle_file_name)


def tag_edit_dist(content_dict, normalizer):
    stats = {"count_0": 0, "count_1": 0, "count_gt_1": 0, "count_lt_1": 0}
    man_text = content_dict["man_text"]
    mach_text = content_dict["mach_text"]

    norm_man_text = normalizer(man_text)
    norm_mach_text = normalizer(mach_text)

    edit_dist = 0.0
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

    if edit_dist == 0.0:
        stats["count_0"] += 1
    elif edit_dist == 1.0:
        stats["count_1"] += 1
    elif edit_dist > 1.0:
        stats["count_gt_1"] += 1
    elif edit_dist < 1.0 and edit_dist > 0.0:
        stats["count_lt_1"] += 1

    return edit_dist, stats


def tag_text_lang(content_dict):
    stats = {"en_count": 0, "non_en_count": 0}
    man_text = content_dict["man_text"]

    *_, details = cld2.detect(man_text)
    lang_id = details[0][1]

    if lang_id == "en":
        stats["en_count"] += 1

    stats["non_en_count"] = 1 - stats["en_count"]

    return lang_id, stats


def tag_casing(content_dict):
    content_iter = content_dict["content_iter"]
    stats = {"count_upper": 0, "count_lower": 0, "count_mixed": 0}

    casing = ""
    casing_dist = {"upper": 0, "lower": 0, "mixed": 0}
    for caption in content_iter:
        if caption.text.strip() != "":
            seen_upper = seen_lower = False
            capset = set(caption.text)
            if not seen_upper and CONST_A2Z.intersection(capset):
                seen_upper = True
            if not seen_lower and CONST_a2z.intersection(capset):
                seen_lower = True

            if seen_upper and seen_lower:
                casing_dist["mixed"] += 1
            elif seen_upper and not seen_lower:
                casing_dist["upper"] += 1
            elif not seen_upper and seen_lower:
                casing_dist["lower"] += 1
        else:
            casing_dist["mixed"] += 1

    max_value = max(casing_dist.values())
    max_keys = [k for k, v in casing_dist.items() if v == max_value]
    if len(max_keys) == 1:
        casing = max_keys[0]
    else:
        if "mixed" in max_keys:
            casing = "mixed"
        else:
            casing = max_keys[0]

    if casing == "upper":
        stats["count_upper"] += 1
    elif casing == "lower":
        stats["count_lower"] += 1
    elif casing == "mixed":
        stats["count_mixed"] += 1

    return casing, stats


def tag_has_comma_period(content_dict):
    content_iter = content_dict["content_iter"]
    stats = {"count": 0}
    has_comma_period = False
    seen_period = seen_comma = False
    for caption in content_iter:
        if not seen_period and "." in caption.text:
            seen_period = True
        if not seen_comma and "," in caption.text:
            seen_comma = True
        if seen_period and seen_comma:
            has_comma_period = True
            stats["count"] += 1
            break

    return has_comma_period, stats


def tag_repeating_lines(content_dict):
    content_iter = content_dict["content_iter"]
    stats = {"count": 0}
    repeating_lines = False
    textset = set()
    for caption in content_iter:
        if caption.text in textset:
            repeating_lines = True
            stats["count"] += 1
            break
        textset.add(caption.text)

    return repeating_lines, stats


def tag_has_proper_cap_after_punct_line(content_dict):
    content_iter = content_dict["content_iter"]
    stats = {"count": 0}
    has_proper_cap_after_punct_line = False

    pattern = r"[.!?](?:\s*)$"
    for i, caption in enumerate(content_iter):
        if i != 0:
            prev_caption = content_iter[i - 1]
            if re.search(pattern, prev_caption.text):
                if caption.text.strip() != "":
                    if not caption.text[0].isupper() and caption.text[0].isalpha():
                        has_proper_cap_after_punct_line = True
                        stats["count"] += 1
                        break

    return has_proper_cap_after_punct_line, stats


TAG_DICT = {
    "has_comma_period": tag_has_comma_period,
    "casing": tag_casing,
    "repeating_lines": tag_repeating_lines,
    "edit_dist": tag_edit_dist,
    "text_lang": tag_text_lang,
    "has_proper_cap_after_punct_line": tag_has_proper_cap_after_punct_line,
}


def process_jsonl(jsonl_path, config_dict, output_dir):
    """Processes a full jsonl file and writes the output processed jsonl file
    (or if the filtering kills all the lines, will output nothing)
    """
    # Read file
    lines = [
        json.loads(_)
        for _ in gzip.decompress(open(jsonl_path, "rb").read()).splitlines()
    ]
    lines_seen = 0
    chars_seen = 0
    dur_seen = 0

    # Process all lines
    output_lines = []
    file_stats = []
    for line in lines:
        lines_seen += 1
        content_dict = {
            "content_iter": parse_into_iter(line["content"], line["subtitle_file"]),
            "man_text": get_man_text(line["content"]),
            "mach_text": (
                get_mach_text(line["mach_content"])
                if line["mach_content"] != ""
                else ""
            ),
        }
        chars_seen += len(line["content"])
        dur_seen += line["length"]
        tags, stats = process_content(content_dict, config_dict)

        for tag, tag_val in tags.items():
            line[tag] = tag_val

        output_lines.append(line)
        file_stats.append(stats)

    cum_file_stats = process_stats(file_stats)
    # Save to output
    output_file = os.path.join(output_dir, os.path.basename(jsonl_path))
    with open(output_file, "wb") as f:
        f.write(
            gzip.compress(
                b"\n".join([json.dumps(_).encode("utf-8") for _ in output_lines])
            )
        )

    return (
        lines_seen,
        chars_seen,
        dur_seen,
        cum_file_stats,
    )


def process_content(content_dict, config) -> Dict[str, Union[str, float, bool]]:
    tags = {}
    stats = {}
    for tag_dict in config["pipeline"]:
        tag_fxn = TAG_DICT[tag_dict["tag"]]
        kwargs = {k: v for k, v in tag_dict.items() if k != "tag"}

        if tag_dict["tag"] == "edit_dist":
            normalizer = EnglishTextNormalizer()
            tag_val, tag_stats = tag_fxn(content_dict, normalizer)
        else:
            tag_val, tag_stats = tag_fxn(content_dict, **kwargs)

        tags[tag_dict["tag"]] = tag_val
        stats[tag_dict["tag"]] = tag_stats

    return tags, stats


def process_stats(stats_list):
    tags = stats_list[0].keys()
    cum_stats = defaultdict(int)
    for tag in tags:
        tag_stats = stats_list[0][tag].keys()
        cum_tag_stats = defaultdict(int)
        for tag_stat in tag_stats:
            cum_tag_stats[tag_stat.replace("count", "avg")] = sum(
                [ele_stats[tag][tag_stat] for ele_stats in stats_list]
            ) / len(stats_list)

        cum_stats[tag] = cum_tag_stats

    return cum_stats


def main(config_path, input_dir, output_dir, num_cpus=None):
    if num_cpus is None:
        num_cpus = os.cpu_count()

    files = glob.glob(f"{input_dir}/*.jsonl.gz")
    os.makedirs(output_dir, exist_ok=True)

    config_dict = parse_config(config_path)
    print("CONFIG IS ", config_dict)
    partial_fxn = partial(process_jsonl, config_dict=config_dict, output_dir=output_dir)
    output_numbers = run_imap_multiprocessing(partial_fxn, files, num_cpus)

    lines_seen, chars_seen, dur_seen, cum_file_stats = zip(*output_numbers)

    cum_process_stats = process_stats(cum_file_stats)

    with open(f"{output_dir}/{config_dict['name']}_cumulative_stats.log", "w") as f:
        f.write(f"Number of lines seen: {sum(lines_seen)}\n")
        f.write(f"Number of characters seen: {sum(chars_seen)}\n")
        f.write(f"Total duration seen: {sum(dur_seen)}\n")
        for tag, tag_stats in cum_process_stats.items():
            f.write(f"{tag}:\n")
            for stat, val in tag_stats.items():
                f.write(f"\t{stat}: {val}\n")
            f.write("\n")


if __name__ == "__main__":
    Fire(main)
