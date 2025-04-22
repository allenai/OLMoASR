import glob
import os
from typing import Tuple, Union, Dict, Any, Literal, Optional, List
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


def process_hitlist(hitlist, pipeline):
    """Processes hitlist in a nice way and prints it out
    Each line looks like:
    Step (int) | Killed %s of total | Killed % of remainder
    """
    total_lines = sum(hitlist.values())
    remainder = total_lines
    max_tag_len = max(len(_["tag"]) for _ in pipeline)

    for i, d in enumerate(pipeline, start=1):
        tag = d["tag"]
        pad = " " * max(0, max_tag_len - len(d["tag"]))
        this_hit = hitlist.get(tag, 0)

        print(
            "Step %02d (%s) %s | Modified %05.02f%% of total | Modified %05.02f%% of remainder"
            % (
                i,
                tag,
                pad,
                100 * this_hit / total_lines,
                100 * this_hit / remainder,
            )
        )

        remainder -= this_hit


def filter_bool(tag_value, ref_value):
    """Filters a boolean value based on the kwargs provided"""
    if tag_value == ref_value:
        return True
    else:
        return False


def filter_cat(tag_value, ref_value, comparison: Optional[Literal["in", "not_in"]] = None):
    """Filters a categorical value based on the kwargs provided"""
    if isinstance(ref_value, str):
        ref_value = [ref_value]

    if comparison is not None:
        if comparison == "in":
            if tag_value in ref_value:
                return True
            else:
                return False
        elif comparison == "not_in":
            if tag_value not in ref_value:
                return True
            else:
                return False
    else:
        if tag_value in ref_value:
            return True
        else:
            return False


def filter_num(tag_value, lower_bound=None, upper_bound=None, inclusive=True):
    valid = set()
    if lower_bound is not None:
        if inclusive:
            if tag_value >= lower_bound:
                valid.add(True)
            else:
                valid.add(False)
        else:
            if tag_value > lower_bound:
                valid.add(True)
            else:
                valid.add(False)

    if upper_bound is not None:
        if inclusive:
            if tag_value <= upper_bound:
                valid.add(True)
            else:
                valid.add(False)
        else:
            if tag_value < upper_bound:
                valid.add(True)
            else:
                valid.add(False)

    if set(valid) == {True}:
        return True
    else:
        return False


# ============================================================
# =                        LOGIC BLOCK                       =
# ============================================================


def process_jsonl(
    jsonl_path,
    output_dir,
    config_dict=None,
    only_subsample=False,
    subsample=False,
    subsample_size=None,
):
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
    subsampled_count = 0
    total_hitlist = {}

    output_lines = []
    if only_subsample is False:
        # Process all lines
        total_hitlist = defaultdict(int)
        for line in lines:
            lines_seen += 1
            chars_seen += len(line["seg_content"])
            output_line, hitlist = process_line(line, config_dict)
            for k, v in hitlist.items():
                total_hitlist[k] += v

            if output_line is not None:
                lines_kept += 1
                chars_kept += len(output_line["seg_content"])
                output_lines.append(line)
            else:
                continue

        # Save to output
        if len(output_lines) > 0:
            if subsample:
                subsample_seed = 42
                output_file = os.path.join(output_dir, os.path.basename(jsonl_path))
                if len(output_lines) > subsample_size:
                    rng = np.random.default_rng(subsample_seed)
                    output_lines = rng.choice(
                        output_lines, size=subsample_size, replace=False
                    )
                subsampled_count = len(output_lines)
            else:
                output_file = os.path.join(output_dir, os.path.basename(jsonl_path))
    else:
        # Only subsample
        subsample_seed = 42
        output_file = os.path.join(output_dir, os.path.basename(jsonl_path))
        if len(lines) > subsample_size:
            rng = np.random.default_rng(subsample_seed)
            output_lines = rng.choice(lines, size=subsample_size, replace=False)
        else:
            output_lines = lines
            
        keys_to_keep = {"id", "seg_id", "subtitle_file", "audio_file", "seg_content"}
        output_lines = [{k: d[k] for k in keys_to_keep if k in d} for d in output_lines]

        subsampled_count = len(output_lines)
        lines_seen = len(lines)

    if len(output_lines) > 0:
        with open(output_file, "wb") as f:
            f.write(
                gzip.compress(
                    b"\n".join([json.dumps(_).encode("utf-8") for _ in output_lines])
                )
            )
    else:
        print(f"{jsonl_path} after filtering resulted in no output lines")

    return (
        lines_seen,
        lines_kept,
        chars_seen,
        chars_kept,
        dict(total_hitlist),
        subsampled_count,
    )


def process_line(line, config):
    hitlist = defaultdict(int)

    keep = True
    for tag_filter_dict in config["pipeline"]:
        tag = tag_filter_dict["tag"]
        kwargs = {k: v for k, v in tag_filter_dict.items() if k != "tag"}

        if isinstance(line[tag], bool):
            keep = filter_bool(line[tag], **kwargs)
        elif isinstance(line[tag], str):
            keep = filter_cat(line[tag], **kwargs)
        elif isinstance(line[tag], int) or isinstance(line[tag], float):
            keep = filter_num(line[tag], **kwargs)

        if keep is False:
            hitlist[tag] += 1
            return None, hitlist

    hitlist["pass"] += 1
    return line, hitlist


# =============================================================
# =                        MAIN BLOCK                         =
# =============================================================


def main(
    input_dir,
    output_dir,
    config_path=None,
    only_subsample=False,
    subsample=False,
    subsample_size=None,
    num_cpus=None,
):
    start_time = time.time()
    if num_cpus == None:
        num_cpus = os.cpu_count()

    files = glob.glob(os.path.join(input_dir, "**/*.jsonl.gz"), recursive=True)

    os.makedirs(output_dir, exist_ok=True)
    if config_path is not None:
        config_dict = parse_config(config_path)
        print("CONFIG IS ", config_dict)
    else:
        config_dict = {}

    partial_fxn = partial(
        process_jsonl,
        output_dir=output_dir,
        config_dict=config_dict,
        only_subsample=only_subsample,
        subsample=subsample,
        subsample_size=subsample_size,
    )
    output_numbers = run_imap_multiprocessing(partial_fxn, files, num_cpus)

    lines_seen = lines_kept = chars_seen = chars_kept = dur_seen = dur_kept = 0
    total_hitlist = defaultdict(int)

    (
        lines_seen_list,
        lines_kept_list,
        chars_seen_list,
        chars_kept_list,
        hitlist_list,
        subsampled_count,
    ) = zip(*output_numbers)
    if only_subsample:
        total_subsampled_count = sum(subsampled_count)
        total_subsampled_dur = (total_subsampled_count * 30) / (60 * 60)

        print(
            "Processed %s files in %.02f seconds"
            % (len(files), time.time() - start_time)
        )

        print(
            "Subsampled %s Lines | Subsampled %.04f Hours | %.04f survival rate"
            % (
                total_subsampled_count,
                total_subsampled_dur,
                100 * (total_subsampled_count / sum(lines_seen_list)),
            )
        )

        with open(
            os.path.join(
                output_dir, "subsampled_stats.log"
            ),
            "w",
        ) as f:
            f.write(
                "Subsampled %s Lines | Subsampled %.04f Hours | %.04f survival rate\n"
                % (
                    total_subsampled_count,
                    total_subsampled_dur,
                    100 * (total_subsampled_count / sum(lines_seen_list)),
                )
            )
    else:
        lines_seen = sum(lines_seen_list)
        lines_kept, mean_lines_kept, med_lines_kept = (
            sum(lines_kept_list),
            np.mean(lines_kept_list),
            np.median(lines_kept_list),
        )
        chars_seen = sum(chars_seen_list)
        chars_kept = sum(chars_kept_list)
        dur_seen = lines_seen * 30 / (60 * 60)
        dur_kept = lines_kept * 30 / (60 * 60)

        for hitlist in hitlist_list:
            for k, v in hitlist.items():
                total_hitlist[k] += v

        if subsample is True:
            total_subsampled_count = sum(subsampled_count)
            total_subsampled_dur = (total_subsampled_count * 30) / (60 * 60)
            print(
                "Subsampled %s Lines | Subsampled %.04f Hours | %.04f survival rate"
                % (
                    total_subsampled_count,
                    total_subsampled_dur,
                    100 * (total_subsampled_count / lines_seen),
                )
            )

        print(
            "Processed %s files in %.02f seconds"
            % (len(files), time.time() - start_time)
        )
        print(
            "Kept %s/%s Lines | %.04f survival rate | Mean %.04f Lines | Median %.04f Lines"
            % (
                lines_kept,
                lines_seen,
                100 * (lines_kept / lines_seen),
                mean_lines_kept,
                med_lines_kept,
            )
        )
        print(
            "Kept %s/%s Chars | %.04f survival rate"
            % (chars_kept, chars_seen, 100 * (chars_kept / chars_seen))
        )
        print(
            "Kept %.04f/%.04f Hours | %.04f survival rate"
            % (dur_kept, dur_seen, 100 * (dur_kept / dur_seen))
        )

        process_hitlist(dict(total_hitlist), config_dict["pipeline"])

        with open(
            os.path.join(
                output_dir, f"{os.path.basename(config_path).split('.yaml')[0]}.log"
            ),
            "w",
        ) as f:
            f.write(
                "Kept %s/%s Lines | %.04f survival rate | Mean %.04f Lines | Median %.04f Lines\n"
                % (
                    lines_kept,
                    lines_seen,
                    100 * (lines_kept / lines_seen),
                    mean_lines_kept,
                    med_lines_kept,
                )
            )
            f.write(
                "Kept %s/%s Chars | %.04f survival rate\n"
                % (chars_kept, chars_seen, 100 * (chars_kept / chars_seen))
            )
            f.write(
                "Kept %.04f/%.04f Hours | %.04f survival rate\n"
                % (
                    dur_kept,
                    dur_seen,
                    100 * (dur_kept / dur_seen),
                )
            )

            if subsample is True:
                f.write(
                    "Subsampled %s Lines | Subsampled %.04f Hours | %.04f survival rate\n"
                    % (
                        total_subsampled_count,
                        total_subsampled_dur,
                        100 * (total_subsampled_count / sum(lines_seen_list)),
                    )
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Add arguments
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
        "--config", type=str, required=False, help="location of the config.yaml"
    )
    parser.add_argument(
        "--only-subsample",
        action="store_true",
        help="Only subsample the data, do not filter it",
    )
    parser.add_argument(
        "--subsample",
        action="store_true",
        help="Subsample the data, do not filter it",
    )
    parser.add_argument(
        "--subsample-size",
        type=int,
        required=False,
        default=1000,
        help="Size of the subsample to take",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        required=False,
        help="How many cpus to process using. Defaults to number of cpus on this machine",
    )
    args = parser.parse_args()

    main(
        args.input_dir,
        args.output_dir,
        config_path=args.config,
        only_subsample=args.only_subsample,
        subsample=args.subsample,
        subsample_size=args.subsample_size,
        num_cpus=args.num_cpus,
    )
