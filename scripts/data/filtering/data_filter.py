import glob
import os
from typing import Tuple, Union, Dict, Any, Literal, Optional
import numpy as np
import io
from collections import defaultdict
import json
import gzip
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import yaml
import time
import argparse
import pysrt
import webvtt
from io import StringIO
import re
import pycld2 as cld2
from open_whisper.utils import TranscriptReader

"""
Single node ec2 data filtering pipeline. 
Just works local2local for now.

Usage will be like:
python jsonl_filter_single_node.py --config filter_config.yaml --input-dir <path/to/local/inputs> --output-dir <path/to/local/outputs>

Example filter config:
'''
name: <name here>
pipeline:
 - fxn: fxn1
   kwarg1: val1
 - fxn: fxn2
   kwarg2: val_a
   kwarg3: val_b
'''



Will take a folder  of jsonl.gz's in and make a folder of jsonl.gz's out
Each line in a jsonl.gz is a json-dict like: 

{'audio_file': '/weka/oe-data-default/huongn/ow_full/00000548/uVC2h48KRos/uVC2h48KRos.m4a',
 'content': 'DUMMY SHORT CONTENT',
 'length': 303.6649375,
 'subtitle_file': '/weka/oe-data-default/huongn/ow_full/00000548/uVC2h48KRos/uVC2h48KRos.en-nP7-2PuUl7o.srt'}


[and only the 'content' key will be modified, but some lines may be deleted!]
-------------------


"""


# =============================================================
# =                          UTILITIES                        =
# =============================================================


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


def parse_man_content(man_content):
    ext = lambda text: "vtt" if text.startswith("WEBVTT") else "srt"
    reader = TranscriptReader(
        file_path=None, transcript_string=man_content, ext=ext(man_content)
    )

    t_dict, *_ = reader.read()
    parsed_content = reader.extract_text(t_dict)

    return parsed_content


def parse_mach_content(mach_content):
    content = webvtt.from_string(mach_content)
    parsed_content_list = []
    for i, caption in enumerate(content):
        if "\n" not in caption.text and i < len(content) - 1:
            parsed_content_list.append(caption.text)
        if i == len(content) - 1:
            parsed_content_list.append(caption.text.split("\n")[-1])

    parsed_content = " ".join(parsed_content_list)

    return parsed_content


def parse_into_iter(content, subtitle_file_name):
    """Parses either the contents of an srt or vtt into an iterable string of things with a .text field"""
    ext = os.path.splitext(subtitle_file_name)[-1]
    if ext == ".srt":
        return pysrt.from_string(content)
    elif ext == ".vtt":
        return webvtt.from_string(content)
    else:
        raise Exception("Unsupported subtitle file type: %s" % subtitle_file_name)


def parse_iter_to_string(iter_contents):
    """Parses the iterable back into the content form it came from (might throw away some vtt metadata)"""
    if isinstance(iter_contents, pysrt.srtfile.SubRipFile):
        sio = StringIO()
        iter_contents.write_into(sio)
        sio.seek(0)
        return sio.read()
    elif isinstance(iter_contents, webvtt.webvtt.WebVTT):
        return iter_contents.content
    else:
        raise Exception("Unknown content type %s" % iter_contents.__class__)


def process_hitlist(hitlist, pipeline):
    """Processes hitlist in a nice way and prints it out
    Each line looks like:
    Step (int) | Killed %s of total | Killed % of remainder
    """
    total_lines = sum(hitlist.values())
    remainder = total_lines
    max_fxn_len = max(len(_["fxn"]) for _ in pipeline)

    for i, d in enumerate(pipeline, start=1):
        fxn = d["fxn"]
        pad = " " * max(0, max_fxn_len - len(d["fxn"]))
        this_hit = hitlist.get(fxn, 0)

        if fxn != "modify_text":
            print(
                "Step %02d (%s) %s | Killed %05.02f%% of total | Killed %05.02f%% of remainder"
                % (
                    i,
                    fxn,
                    pad,
                    100 * this_hit / total_lines,
                    100 * this_hit / remainder,
                )
            )
        else:
            print(
                "Step %02d (%s) %s | Modified %05.02f%% of total | Modified %05.02f%% of remainder"
                % (
                    i,
                    fxn,
                    pad,
                    100 * this_hit / total_lines,
                    100 * this_hit / remainder,
                )
            )
        remainder -= this_hit


# =============================================================
# =                         FILTERING ATOMS                   =
# =============================================================
# Some constants
CONST_a2z = set([chr(ord("a") + i) for i in range(26)])
CONST_A2Z = set(_.upper() for _ in CONST_a2z)


def _filter_stub(content, **kwargs):
    """Basic filtering stub. Always takes in a content and some kwargs
    Content is of type SubRipFile or WebVTT. Both have iterables that have .text fields

    Will either return:
        None (kill the whole line)
    OR
        the new content (of the same type, but potentially different text)
    """
    if content == None:
        return None


def identity_filter(content):
    # Just a dummy identity map, aw
    return content


# text_heurs_1 filters
def has_comma_period(content):
    # Returns full content if both a ',' and '.' are contained in the content
    seen_period = seen_comma = False
    for caption in content:
        if not seen_period and "." in caption.text:
            seen_period = True
        if not seen_comma and "," in caption.text:
            seen_comma = True
        if seen_period and seen_comma:
            return content

    return None


def has_mixed_case(content):
    # Returns full content if both an uppercase and lowercase character are present
    seen_upper = seen_lower = False

    for caption in content:
        capset = set(caption.text)
        if not seen_upper and CONST_A2Z.intersection(capset):
            seen_upper = True
        if not seen_lower and CONST_a2z.intersection(capset):
            seen_lower = True
        if seen_upper and seen_lower:
            return content

    return None


def has_no_repeats(content):
    textset = set()
    for caption in content:
        if caption.text in textset:
            return None
        textset.add(caption.text)

    return content


def modify_text(content):
    # Pattern to match brackets containing capitalized words, excluding the word "Music"
    pattern_brackets = (
        r"[ ]*\[(?![Mm][Uu][Ss][Ii][Cc]\])([A-Z][a-zA-Z]*(?: [A-Z][a-zA-Z]*)*)\][ ]*"
    )

    # Pattern to match parentheses containing any characters
    pattern_parentheses = r"[ ]*\(.*?\)[ ]*"

    # Pattern to match capitalized words followed by a colon
    pattern_colon = r"[ ]*(?:[A-Z][a-zA-Z]*[ ])+:[ ]*"

    # Pattern to match specific strings like &nbsp;, &gt;, =, and ...
    specific_strings = r"[ ]*(?:&nbsp;|&amp;|&lt;|&gt;|=|\.{3}|\\h)+[ ]*"

    # Combined primary pattern using the above patterns
    primary_pattern = (
        f"{pattern_brackets}|{pattern_parentheses}|{pattern_colon}|{specific_strings}"
    )

    # Pattern to capture lowercase words inside brackets
    # brackets_pattern_capture = r"\[([a-z]+(?: [a-z]+)*)\]"

    mod_count = 0
    for caption in content:
        new_text = re.sub(primary_pattern, " ", caption.text)
        if new_text != caption.text:
            mod_count += 1
        caption.text = new_text
        # caption.text = re.sub(brackets_pattern_capture, r"\1", caption.text)

    return content, mod_count


# manmach filter
def filter_unrelated(scores_dict: Dict, threshold: int, comparison: str):
    score = scores_dict["man_mach_score"]

    if comparison == "ge":
        return score >= threshold
    elif comparison == "g":
        return score > threshold
    elif comparison == "le":
        return score <= threshold
    elif comparison == "l":
        return score < threshold


def filter_bad_align_edit_dist(scores_dict: Dict, threshold: int, comparison: str):
    edit_dist = scores_dict["edit_dist"]

    if comparison == "ge":
        return edit_dist >= threshold
    elif comparison == "g":
        return edit_dist > threshold
    elif comparison == "le":
        return edit_dist <= threshold
    elif comparison == "l":
        return edit_dist < threshold


# text_heurs_2 filters
def has_proper_capitalization_and_punctuation(content):
    # Sentence-ending punctuation followed by a lowercase letter, Punctuation surrounded by whitespace, Punctuation preceded by whitespace
    pattern = r".[.?!]\s+[a-z]|\s[.,;!?]\s"

    for caption in content:
        if re.search(pattern, caption.text):
            return None
    return content


def has_proper_capitalization_after_punctuation_line(content):
    pattern = r"[.!?](?:\s*)$"
    for i, caption in enumerate(content):
        if i != 0:
            prev_caption = content[i - 1]
            if re.search(pattern, prev_caption.text):
                if caption.text.strip() != "":
                    if not caption.text[0].isupper() and caption.text[0].isalpha():
                        return None
    return content

def empty_caption(content):
    for caption in content:
        if caption.text.strip() == "":
            return None
    return content


# lang_align filter (en only)
def lang_align(lang_dict):
    audio_lang = lang_dict["audio_lang"]
    text_lang = lang_dict["text_lang"]
    if audio_lang == "en" and text_lang == "en":
        return True
    return False


# text_heurs_3 filters
def consecutive_words_starting_with_upper(
    content,
    word_level_threshold,
    transcript_level_threshold,
    consecutive_threshold,
    consecutive_bad_threshold,
):
    bad_captions = 0
    consecutive_freq = 0
    consecutive_bad_captions = 0
    occurred = False
    for caption in content:
        if caption.text.strip() != "":
            words = caption.text.split()
            word_rate = sum([1 for word in words if word[0].isupper()]) / len(words)
            if not occurred and consecutive_freq >= consecutive_threshold:
                consecutive_bad_captions += 1
                occurred = True

            if word_rate > word_level_threshold:
                consecutive_freq += 1
                bad_captions += 1
            else:
                consecutive_freq = 0
                occurred = False
        if (
            bad_captions > transcript_level_threshold
            and consecutive_bad_captions > consecutive_bad_threshold
        ):
            return None
        return content


def consecutive_caption_starting_with_upper(
    content, caption_level_threshold, transcript_level_threshold
):
    consecutive_freq = 0
    consecutive_transcript_freq = 0
    occurred = False
    for caption in content:
        if not occurred and consecutive_freq >= caption_level_threshold:
            consecutive_transcript_freq += 1
            occurred = True

        if caption.text[0].isupper():
            consecutive_freq += 1
        else:
            consecutive_freq = 0
            occurred = False

    if consecutive_transcript_freq > transcript_level_threshold:
        return None
    return content


FILTER_DICT = {
    "identity": identity_filter,
    "has_comma_period": has_comma_period,
    "has_mixed_case": has_mixed_case,
    "has_no_repeats": has_no_repeats,
    "modify_text": modify_text,
    "filter_unrelated": filter_unrelated,
    "has_proper_capitalization_and_punctuation": has_proper_capitalization_and_punctuation,
    "has_proper_capitalization_after_punctuation_line": has_proper_capitalization_after_punctuation_line,
    "empty_caption": empty_caption,
    "filter_bad_align_edit_dist": filter_bad_align_edit_dist,
    "lang_align": lang_align,
}


# ============================================================
# =                        LOGIC BLOCK                       =
# ============================================================


def process_jsonl(jsonl_path, config_dict, output_dir):
    """Processes a full jsonl file and writes the output processed jsonl file
    (or if the filtering kills all the lines, will output nothing)
    """
    # Read file
    lines = [
        json.loads(_)
        for _ in gzip.decompress(open(jsonl_path, "rb").read()).splitlines()
    ]
    lines_seen = lines_kept = 0
    chars_seen = chars_kept = 0
    dur_seen = dur_kept = 0

    # Process all lines
    output_lines = []
    total_hitlist = defaultdict(int)
    for line in lines:
        lines_seen += 1
        parsed_content = parse_into_iter(line["content"], line["subtitle_file"])
        if "audio_lang" in line.keys():
            lang_dict = {
                "audio_lang": line["audio_lang"],
                "text_lang": line["text_lang"]
            }
        scores_dict = {
            k: v for k, v in line.items() if "score" in k or "edit_dist" in k
        }
        chars_seen += len(line["content"])  # TODO: Be more precise here
        dur_seen += line["length"]
        output_content, hitlist = process_content(
            parsed_content, scores_dict, lang_dict, config_dict
        )
        for k, v in hitlist.items():
            total_hitlist[k] += v

        if output_content != None:
            output_content_str = parse_iter_to_string(output_content)
            lines_kept += 1
            chars_kept += len(output_content_str)  # TODO: Be more precise here
            dur_kept += line["length"]
            line["content"] = output_content_str
            output_lines.append(line)
        else:
            continue

    # Save to output
    if len(output_lines) > 0:
        output_file = os.path.join(output_dir, os.path.basename(jsonl_path))
        with open(output_file, "wb") as f:
            f.write(
                gzip.compress(
                    b"\n".join([json.dumps(_).encode("utf-8") for _ in output_lines])
                )
            )

    return (
        lines_seen,
        lines_kept,
        chars_seen,
        chars_kept,
        dur_seen,
        dur_kept,
        dict(total_hitlist),
    )


def process_content(content, scores_dict, lang_dict: Optional[Dict], config):
    hitlist = defaultdict(int)
    unrelated_keep = []
    for filter_dict in config["pipeline"]:
        filter_fxn = FILTER_DICT[filter_dict["fxn"]]
        kwargs = {k: v for k, v in filter_dict.items() if k != "fxn"}

        if filter_dict["fxn"] == "filter_unrelated":
            keep = filter_fxn(scores_dict, **kwargs)
            unrelated_keep.append(keep)
            # if not keep:
            #     content = None
        elif filter_dict["fxn"] == "filter_bad_align_edit_dist":
            keep = filter_fxn(scores_dict, **kwargs)
            if not keep:
                content = None
        elif filter_dict["fxn"] == "modify_text":
            content, mod_count = filter_fxn(content, **kwargs)
            if mod_count > 0:
                hitlist[filter_dict["fxn"]] += 1
        elif filter_dict["fxn"] == "lang_align":
            keep = filter_fxn(lang_dict)
            if not keep:
                content = None
        else:
            content = filter_fxn(content, **kwargs)

        if content == None:
            hitlist[filter_dict["fxn"]] += 1
            return None, hitlist

    # multiple comparisons for filter_unrelated
    if content is not None and len(unrelated_keep) > 0:
        if False in set(unrelated_keep):
            hitlist["filter_unrelated"] += 1
            return None, hitlist
        elif set(unrelated_keep) == {True}:
            pass

    hitlist["pass"] += 1
    return content, hitlist


# =============================================================
# =                        MAIN BLOCK                         =
# =============================================================


def main(config_path, input_dir, output_dir, num_cpus=None):
    start_time = time.time()
    if num_cpus == None:
        num_cpus = os.cpu_count()

    files = glob.glob(os.path.join(input_dir, "**/*.jsonl.gz"), recursive=True)

    os.makedirs(output_dir, exist_ok=True)
    config_dict = parse_config(config_path)
    print("CONFIG IS ", config_dict)
    partial_fxn = partial(process_jsonl, config_dict=config_dict, output_dir=output_dir)
    output_numbers = run_imap_multiprocessing(partial_fxn, files, num_cpus)

    lines_seen = lines_kept = chars_seen = chars_kept = dur_seen = dur_kept = 0
    total_hitlist = defaultdict(int)
    for ls, lk, cs, ck, ds, dk, hitlist in output_numbers:
        lines_seen += ls
        lines_kept += lk
        chars_seen += cs
        chars_kept += ck
        dur_seen += ds
        dur_kept += dk
        for k, v in hitlist.items():
            total_hitlist[k] += v

    print(
        "Processed %s files in %.02f seconds" % (len(files), time.time() - start_time)
    )
    print(
        "Kept %s/%s Lines | %.04f survival rate"
        % (lines_kept, lines_seen, 100 * (lines_kept / lines_seen))
    )
    print(
        "Kept %s/%s Chars | %.04f survival rate"
        % (chars_kept, chars_seen, 100 * (chars_kept / chars_seen))
    )
    print(
        "Kept %.04f/%.04f Hours | %.04f survival rate"
        % (dur_kept / (60 * 60), dur_seen / (60 * 60), 100 * (dur_kept / dur_seen))
    )

    process_hitlist(dict(total_hitlist), config_dict["pipeline"])

    with open(
        os.path.join(
            output_dir, f"{os.path.basename(config_path).split('.yaml')[0]}.log"
        ),
        "w",
    ) as f:
        f.write(
            "Kept %s/%s Lines | %.04f survival rate\n"
            % (lines_kept, lines_seen, 100 * (lines_kept / lines_seen))
        )
        f.write(
            "Kept %s/%s Chars | %.04f survival rate\n"
            % (chars_kept, chars_seen, 100 * (chars_kept / chars_seen))
        )
        f.write(
            "Kept %.04f/%.04f Hours | %.04f survival rate\n"
            % (dur_kept / (60 * 60), dur_seen / (60 * 60), 100 * (dur_kept / dur_seen))
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Add arguments
    parser.add_argument(
        "--config", type=str, required=True, help="location of the config.yaml"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="location of the input jsonl.gz files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="location of where the output jsonl.gz files will go",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        required=False,
        help="How many cpus to process using. Defaults to number of cpus on this machine",
    )
    args = parser.parse_args()

    main(args.config, args.input_dir, args.output_dir, num_cpus=args.num_cpus)
