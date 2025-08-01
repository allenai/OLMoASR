"""
Tagged data processing pipeline for JSONL subtitle files.

This module provides a comprehensive filtering and processing system for tagged
subtitle data stored in JSONL format. It supports various filtering operations
including boolean, categorical, and numeric filters, as well as text modification
and subsampling capabilities.

Features:
- Multi-type filtering (boolean, categorical, numeric)
- Text content modification with regex patterns
- Parallel processing for large datasets
- Statistical reporting and survival rate analysis
- Configurable filtering pipelines via YAML
- Data subsampling with optional filtering
- Progress tracking and detailed logging

The pipeline processes tagged data through multiple filtering steps, applying
each filter in sequence and tracking statistics about data survival rates
and modifications at each step.
"""

import argparse
import glob
import gzip
import json
import os
import re
import time
from collections import defaultdict
from functools import partial
from io import StringIO
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import yaml
from tqdm import tqdm

# Optional imports with fallback handling
try:
    import pycld2 as cld2  # type: ignore

    CLD2_AVAILABLE = True
except ImportError:
    CLD2_AVAILABLE = False
    cld2 = None

try:
    from olmoasr.utils import TranscriptReader  # type: ignore

    OLMOASR_AVAILABLE = True
except ImportError:
    OLMOASR_AVAILABLE = False
    TranscriptReader = None

try:
    import webvtt  # type: ignore

    WEBVTT_AVAILABLE = True
except ImportError:
    WEBVTT_AVAILABLE = False
    webvtt = None

# Type aliases for better readability
FilePath = str
JSONValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
FilterResult = bool
HitList = Dict[str, int]
ProcessingStats = Tuple[int, int, int, int, HitList, int]
ConfigDict = Dict[str, Any]
SubtitleContent = Union[str, Any]


def run_parallel_processing(
    func: callable, argument_list: List[Any], num_processes: int
) -> List[Any]:
    """
    Execute a function in parallel with progress tracking.

    Args:
        func: Function to execute in parallel
        argument_list: List of arguments to pass to the function
        num_processes: Number of parallel processes to use

    Returns:
        List of results from parallel execution

    Raises:
        ValueError: If num_processes is invalid
    """
    if num_processes <= 0:
        raise ValueError("num_processes must be positive")

    with Pool(processes=num_processes) as pool:
        results = []
        for result in tqdm(
            pool.imap(func=func, iterable=argument_list),
            total=len(argument_list),
            desc="Processing files",
        ):
            results.append(result)

    return results


def load_config(config_path: FilePath) -> ConfigDict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return config_dict
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML configuration: {e}")


def display_hitlist_report(hitlist: HitList, pipeline: List[Dict[str, Any]]) -> None:
    """
    Display a formatted report of filtering results.

    This function generates a step-by-step report showing how many items
    were modified or filtered at each stage of the pipeline, including
    percentages relative to the total and remaining items.

    Args:
        hitlist: Dictionary mapping filter names to hit counts
        pipeline: List of pipeline step configurations

    Example Output:
        Step 01 (has_comma_period) | Modified 05.23% of total | Modified 05.23% of remainder
        Step 02 (casing)           | Modified 12.45% of total | Modified 13.15% of remainder
    """
    if not hitlist or not pipeline:
        return

    total_lines = sum(hitlist.values())
    remainder = total_lines
    max_tag_length = max(len(step["tag"]) for step in pipeline)

    print("\nFiltering Results:")
    print("=" * 80)

    for i, step_config in enumerate(pipeline, start=1):
        tag = step_config["tag"]
        padding = " " * max(0, max_tag_length - len(tag))
        hit_count = hitlist.get(tag, 0)

        if total_lines > 0:
            total_percentage = 100 * hit_count / total_lines
            remainder_percentage = 100 * hit_count / remainder if remainder > 0 else 0.0
        else:
            total_percentage = remainder_percentage = 0.0

        print(
            f"Step {i:02d} ({tag}) {padding} | "
            f"Modified {total_percentage:05.02f}% of total | "
            f"Modified {remainder_percentage:05.02f}% of remainder"
        )

        remainder = max(0, remainder - hit_count)


def parse_subtitle_content(transcript_string: str) -> SubtitleContent:
    """
    Parse subtitle content based on format detection.

    Args:
        transcript_string: Raw subtitle content string

    Returns:
        Parsed subtitle object (WebVTT or SRT)

    Raises:
        ImportError: If required subtitle parsing libraries are not available
        ValueError: If content cannot be parsed
    """
    try:
        if transcript_string.startswith("WEBVTT"):
            if not WEBVTT_AVAILABLE or webvtt is None:
                raise ImportError("webvtt library is required but not available")
            return webvtt.from_string(transcript_string)
    except Exception as e:
        raise ValueError(f"Failed to parse subtitle content: {e}")


def serialize_subtitle_content(subtitle_content: SubtitleContent) -> str:
    """
    Convert parsed subtitle content back to string format.

    Args:
        subtitle_content: Parsed subtitle object (WebVTT or SRT)

    Returns:
        Serialized subtitle content as string

    Raises:
        ValueError: If subtitle content type is not supported
    """
    try:
        if WEBVTT_AVAILABLE and hasattr(subtitle_content, "content"):
            # Handle WebVTT content
            return subtitle_content.content
        else:
            raise ValueError("Unsupported subtitle content type")
    except Exception as e:
        raise ValueError(f"Failed to serialize subtitle content: {e}")


def apply_boolean_filter(tag_value: bool, reference_value: bool) -> FilterResult:
    """
    Apply boolean filtering logic.

    Args:
        tag_value: The boolean value to test
        reference_value: The expected boolean value

    Returns:
        True if the filter passes, False otherwise
    """
    return tag_value == reference_value


def apply_categorical_filter(
    tag_value: str,
    reference_value: Union[str, List[str]],
    comparison: Optional[Literal["in", "not_in"]] = None,
) -> FilterResult:
    """
    Apply categorical filtering logic.

    Args:
        tag_value: The categorical value to test
        reference_value: Single value or list of values to compare against
        comparison: Type of comparison ("in", "not_in", or None for default "in")

    Returns:
        True if the filter passes, False otherwise
    """
    # Normalize reference value to list
    if isinstance(reference_value, str):
        reference_values = [reference_value]
    else:
        reference_values = reference_value

    # Apply comparison logic
    if comparison == "not_in":
        return tag_value not in reference_values
    else:  # Default to "in" behavior
        return tag_value in reference_values


def apply_numeric_filter(
    tag_value: Union[int, float],
    lower_bound: Optional[Union[int, float]] = None,
    upper_bound: Optional[Union[int, float]] = None,
    inclusive: bool = True,
) -> FilterResult:
    """
    Apply numeric filtering logic with bounds checking.

    Args:
        tag_value: The numeric value to test
        lower_bound: Optional minimum value (inclusive or exclusive based on inclusive param)
        upper_bound: Optional maximum value (inclusive or exclusive based on inclusive param)
        inclusive: Whether bounds are inclusive (default: True)

    Returns:
        True if the filter passes, False otherwise
    """
    conditions = []

    # Check lower bound
    if lower_bound is not None:
        if inclusive:
            conditions.append(tag_value >= lower_bound)
        else:
            conditions.append(tag_value > lower_bound)

    # Check upper bound
    if upper_bound is not None:
        if inclusive:
            conditions.append(tag_value <= upper_bound)
        else:
            conditions.append(tag_value < upper_bound)

    # All conditions must be true
    return all(conditions) if conditions else True


def clean_subtitle_text(transcript_string: str) -> Tuple[str, int]:
    """
    Clean and modify subtitle text content using regex patterns.

    This function removes common subtitle artifacts including:
    - Speaker names (capitalized words followed by colons)
    - HTML entities (&nbsp;, &amp;, &lt;, &gt;)
    - Special characters (=, ellipsis, \\h)

    Args:
        transcript_string: Raw subtitle content string

    Returns:
        Tuple of (cleaned_transcript_string, modification_count)
        modification_count is 1 if any changes were made, 0 otherwise

    Raises:
        ValueError: If subtitle content cannot be parsed or processed
    """
    try:
        captions = parse_subtitle_content(transcript_string)

        # Define cleaning patterns
        speaker_pattern = r"[ ]*(?:[A-Z][a-zA-Z]*[ ])+:[ ]*"
        html_entities = r"[ ]*(?:&nbsp;|&amp;|&lt;|&gt;|=|\.{3}|\\h)+[ ]*"
        combined_pattern = f"{speaker_pattern}|{html_entities}"

        modification_count = 0

        # Apply cleaning to each caption
        for caption in captions:
            original_text = caption.text
            cleaned_text = re.sub(combined_pattern, " ", original_text)

            if cleaned_text != original_text:
                modification_count = 1
                caption.text = cleaned_text

        # Convert back to string format
        cleaned_transcript = serialize_subtitle_content(captions)
        return cleaned_transcript, modification_count

    except Exception as e:
        raise ValueError(f"Failed to clean subtitle text: {e}")


def process_single_jsonl_file(
    jsonl_path: FilePath,
    output_dir: FilePath,
    config_dict: Optional[ConfigDict] = None,
    only_subsample: bool = False,
    subsample: bool = False,
    subsample_size: Optional[int] = None,
) -> ProcessingStats:
    """
    Process a single JSONL file with filtering and optional subsampling.

    Args:
        jsonl_path: Path to the input JSONL.gz file
        output_dir: Directory to save processed results
        config_dict: Configuration dictionary with filtering pipeline
        only_subsample: If True, only subsample without filtering
        subsample: If True, subsample after filtering
        subsample_size: Number of samples to keep in subsampling

    Returns:
        Tuple of processing statistics:
        (lines_seen, lines_kept, chars_seen, chars_kept, hitlist, subsampled_count)

    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
        ValueError: If processing fails
    """
    # Load and validate input data
    try:
        with open(jsonl_path, "rb") as f:
            compressed_data = f.read()

        decompressed_data = gzip.decompress(compressed_data)
        lines = [
            json.loads(line.decode("utf-8").strip())
            for line in decompressed_data.splitlines()
            if line.strip()
        ]
    except Exception as e:
        raise ValueError(f"Failed to load JSONL file {jsonl_path}: {e}")

    # Initialize counters
    lines_seen = len(lines)
    lines_kept = 0
    chars_seen = sum(len(line.get("seg_content", "")) for line in lines)
    chars_kept = 0
    subsampled_count = 0
    total_hitlist = defaultdict(int)

    if only_subsample:
        # Only perform subsampling without filtering
        output_lines = _perform_subsampling(lines, subsample_size)
        subsampled_count = len(output_lines)

        # Keep only essential keys for subsampled data
        essential_keys = {
            "id",
            "seg_id",
            "subtitle_file",
            "audio_file",
            "seg_content",
            "ts_mode",
            "only_no_ts_mode",
            "norm_end",
        }
        output_lines = [
            {key: line[key] for key in essential_keys if key in line}
            for line in output_lines
        ]
    else:
        # Apply filtering pipeline
        output_lines = []

        for line in lines:
            processed_line, hitlist = _process_line_with_filters(line, config_dict)

            # Update hit statistics
            for tag, count in hitlist.items():
                total_hitlist[tag] += count

            if processed_line is not None:
                lines_kept += 1
                chars_kept += len(processed_line.get("seg_content", ""))
                output_lines.append(processed_line)

        # Apply post-filtering subsampling if requested
        if subsample and output_lines:
            output_lines = _perform_subsampling(output_lines, subsample_size)
            subsampled_count = len(output_lines)

    # Save results if there are lines to output
    if output_lines:
        _save_processed_lines(output_lines, jsonl_path, output_dir)
    else:
        print(f"Warning: {jsonl_path} resulted in no output lines after processing")

    return (
        lines_seen,
        lines_kept,
        chars_seen,
        chars_kept,
        dict(total_hitlist),
        subsampled_count,
    )


def _perform_subsampling(
    lines: List[Dict[str, Any]], subsample_size: Optional[int]
) -> List[Dict[str, Any]]:
    """
    Perform subsampling on a list of lines.

    Args:
        lines: List of data lines to subsample
        subsample_size: Number of samples to keep (None for no subsampling)

    Returns:
        Subsampled list of lines
    """
    if subsample_size is None or len(lines) <= subsample_size:
        return lines

    # Use fixed seed for reproducible results
    rng = np.random.default_rng(42)
    return rng.choice(lines, size=subsample_size, replace=False).tolist()


def _process_line_with_filters(
    line: Dict[str, Any], config: Optional[ConfigDict]
) -> Tuple[Optional[Dict[str, Any]], HitList]:
    """
    Process a single line through the filtering pipeline.

    Args:
        line: Data line to process
        config: Configuration dictionary with pipeline settings

    Returns:
        Tuple of (processed_line_or_None, hitlist)
        processed_line_or_None is None if line was filtered out
    """
    hitlist = defaultdict(int)

    if not config or "pipeline" not in config:
        hitlist["pass"] += 1
        return line, dict(hitlist)

    # Process each filter in the pipeline
    for filter_config in config["pipeline"]:
        tag = filter_config["tag"]
        filter_kwargs = {k: v for k, v in filter_config.items() if k != "tag"}

        # Handle special text content modification
        if tag == "seg_content":
            if tag in line:
                cleaned_content, mod_count = clean_subtitle_text(line[tag])
                line[tag] = cleaned_content
                # Note: modification count could be tracked in hitlist if needed
            continue

        # Apply appropriate filter based on value type
        if tag not in line:
            continue  # Skip if tag not present in line

        tag_value = line[tag]
        filter_passed = True

        try:
            if isinstance(tag_value, bool):
                filter_passed = apply_boolean_filter(tag_value, **filter_kwargs)
            elif isinstance(tag_value, str) and tag != "seg_content":
                filter_passed = apply_categorical_filter(tag_value, **filter_kwargs)
            elif isinstance(tag_value, (int, float)):
                filter_passed = apply_numeric_filter(tag_value, **filter_kwargs)
        except Exception as e:
            print(f"Warning: Filter '{tag}' failed with error: {e}")
            filter_passed = True  # Continue processing on filter errors

        # If filter failed, record hit and return None
        if not filter_passed:
            hitlist[tag] += 1
            return None, dict(hitlist)

    # All filters passed
    hitlist["pass"] += 1
    return line, dict(hitlist)


def _save_processed_lines(
    output_lines: List[Dict[str, Any]], original_path: FilePath, output_dir: FilePath
) -> None:
    """
    Save processed lines to a compressed JSONL file.

    Args:
        output_lines: List of processed data lines
        original_path: Original file path (used for naming output)
        output_dir: Directory to save the output file

    Raises:
        IOError: If file cannot be written
    """
    try:
        output_filename = os.path.basename(original_path)
        output_path = os.path.join(output_dir, output_filename)

        # Serialize lines to JSON and compress
        json_lines = [json.dumps(line).encode("utf-8") for line in output_lines]
        compressed_data = gzip.compress(b"\n".join(json_lines))

        # Write compressed data
        with open(output_path, "wb") as f:
            f.write(compressed_data)

    except Exception as e:
        raise IOError(f"Failed to save processed lines to {output_path}: {e}")


def generate_processing_report(
    files_processed: int,
    processing_time: float,
    lines_stats: Tuple[int, int],
    chars_stats: Tuple[int, int],
    duration_stats: Tuple[float, float],
    hitlist: HitList,
    pipeline: List[Dict[str, Any]],
    subsample_stats: Optional[Tuple[int, float]] = None,
) -> str:
    """
    Generate a comprehensive processing report.

    Args:
        files_processed: Number of files processed
        processing_time: Total processing time in seconds
        lines_stats: Tuple of (lines_seen, lines_kept)
        chars_stats: Tuple of (chars_seen, chars_kept)
        duration_stats: Tuple of (duration_seen, duration_kept) in hours
        hitlist: Dictionary of filter hit counts
        pipeline: Pipeline configuration
        subsample_stats: Optional tuple of (subsampled_count, subsampled_duration)

    Returns:
        Formatted report string
    """
    lines_seen, lines_kept = lines_stats
    chars_seen, chars_kept = chars_stats
    duration_seen, duration_kept = duration_stats

    lines_survival = (100 * lines_kept / lines_seen) if lines_seen > 0 else 0.0
    chars_survival = (100 * chars_kept / chars_seen) if chars_seen > 0 else 0.0
    duration_survival = (
        (100 * duration_kept / duration_seen) if duration_seen > 0 else 0.0
    )

    report_lines = [
        "=" * 80,
        "PROCESSING SUMMARY REPORT",
        "=" * 80,
        f"Files processed: {files_processed}",
        f"Processing time: {processing_time:.2f} seconds",
        "",
        "DATA STATISTICS:",
        f"  Lines: {lines_kept:,}/{lines_seen:,} ({lines_survival:.4f}% survival rate)",
        f"  Characters: {chars_kept:,}/{chars_seen:,} ({chars_survival:.4f}% survival rate)",
        f"  Duration: {duration_kept:.4f}/{duration_seen:.4f} hours ({duration_survival:.4f}% survival rate)",
    ]

    if subsample_stats:
        subsampled_count, subsampled_duration = subsample_stats
        subsample_survival = (
            (100 * subsampled_count / lines_seen) if lines_seen > 0 else 0.0
        )
        report_lines.extend(
            [
                "",
                "SUBSAMPLING STATISTICS:",
                f"  Subsampled: {subsampled_count:,} lines ({subsampled_duration:.4f} hours)",
                f"  Subsample rate: {subsample_survival:.4f}%",
            ]
        )

    return "\n".join(report_lines)


def save_processing_log(
    report: str,
    output_dir: FilePath,
    config_path: Optional[FilePath] = None,
    is_subsample_only: bool = False,
) -> None:
    """
    Save processing report to a log file.

    Args:
        report: Report content to save
        output_dir: Directory to save the log file
        config_path: Optional path to config file (used for naming)
        is_subsample_only: Whether this was a subsample-only operation

    Raises:
        IOError: If log file cannot be written
    """
    try:
        if is_subsample_only:
            log_filename = "subsampled_stats.log"
        elif config_path:
            config_basename = os.path.basename(config_path).replace(".yaml", "")
            log_filename = f"{config_basename}.log"
        else:
            log_filename = "processing_stats.log"

        log_path = os.path.join(output_dir, log_filename)

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(report)

    except Exception as e:
        raise IOError(f"Failed to save processing log: {e}")


def main(
    input_dir: FilePath,
    output_dir: FilePath,
    config_path: Optional[FilePath] = None,
    only_subsample: bool = False,
    subsample: bool = False,
    subsample_size: Optional[int] = None,
    num_cpus: Optional[int] = None,
) -> None:
    """
    Main entry point for tagged data processing pipeline.

    This function orchestrates the entire processing workflow:
    1. Discovers JSONL files in the input directory
    2. Loads configuration if provided
    3. Processes files in parallel with filtering and/or subsampling
    4. Generates comprehensive statistics and reports
    5. Saves results and logs

    Args:
        input_dir: Directory containing input JSONL.gz files
        output_dir: Directory to save processed results
        config_path: Optional path to YAML configuration file
        only_subsample: If True, only subsample without filtering
        subsample: If True, subsample after filtering
        subsample_size: Number of samples to keep (default: 1000)
        num_cpus: Number of CPU cores to use (default: all available)

    Raises:
        FileNotFoundError: If input directory or config file doesn't exist
        ValueError: If invalid parameters are provided
        RuntimeError: If processing fails
    """
    start_time = time.time()

    # Validate inputs
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if config_path and not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Set defaults
    if num_cpus is None:
        num_cpus = os.cpu_count() or 1

    if subsample_size is None:
        subsample_size = 1000

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Discover input files
    file_pattern = os.path.join(input_dir, "**/*.jsonl.gz")
    files = glob.glob(file_pattern, recursive=True)

    if not files:
        raise FileNotFoundError(f"No JSONL.gz files found in {input_dir}")

    # Load configuration
    config_dict = {}
    if config_path and not only_subsample:
        config_dict = load_config(config_path)
        print(f"Loaded configuration: {config_dict}")

    print(f"Processing {len(files)} files with {num_cpus} CPU cores")

    # Create partial function for parallel processing
    process_function = partial(
        process_single_jsonl_file,
        output_dir=output_dir,
        config_dict=config_dict,
        only_subsample=only_subsample,
        subsample=subsample,
        subsample_size=subsample_size,
    )

    # Process files in parallel
    processing_results = run_parallel_processing(process_function, files, num_cpus)

    # Aggregate results
    processing_time = time.time() - start_time
    (
        lines_seen_list,
        lines_kept_list,
        chars_seen_list,
        chars_kept_list,
        hitlist_list,
        subsampled_count_list,
    ) = zip(*processing_results)

    # Calculate aggregate statistics
    total_lines_seen = sum(lines_seen_list)
    total_lines_kept = sum(lines_kept_list)
    total_chars_seen = sum(chars_seen_list)
    total_chars_kept = sum(chars_kept_list)
    total_subsampled = sum(subsampled_count_list)

    # Convert to duration (assuming 30 seconds per line)
    duration_seen = total_lines_seen * 30 / 3600  # hours
    duration_kept = total_lines_kept * 30 / 3600  # hours
    subsampled_duration = total_subsampled * 30 / 3600  # hours

    # Aggregate hitlist
    total_hitlist = defaultdict(int)
    for hitlist in hitlist_list:
        for tag, count in hitlist.items():
            total_hitlist[tag] += count

    # Generate and display report
    if only_subsample:
        subsample_survival = (
            (100 * total_subsampled / total_lines_seen) if total_lines_seen > 0 else 0.0
        )
        print(f"\nSubsampling completed in {processing_time:.2f} seconds")
        print(
            f"Subsampled {total_subsampled:,} lines ({subsampled_duration:.4f} hours)"
        )
        print(f"Subsample survival rate: {subsample_survival:.4f}%")

        # Save subsample log
        report = (
            f"Subsampled {total_subsampled:,} lines | "
            f"Duration: {subsampled_duration:.4f} hours | "
            f"Survival rate: {subsample_survival:.4f}%\n"
        )
        save_processing_log(report, output_dir, is_subsample_only=True)
    else:
        # Display detailed filtering results
        mean_lines_kept = np.mean(lines_kept_list)
        median_lines_kept = np.median(lines_kept_list)

        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        print(f"Files processed: {len(files)}")

        lines_survival = (
            (100 * total_lines_kept / total_lines_seen) if total_lines_seen > 0 else 0.0
        )
        chars_survival = (
            (100 * total_chars_kept / total_chars_seen) if total_chars_seen > 0 else 0.0
        )
        duration_survival = (
            (100 * duration_kept / duration_seen) if duration_seen > 0 else 0.0
        )

        print(
            f"Lines: {total_lines_kept:,}/{total_lines_seen:,} "
            f"({lines_survival:.4f}% survival) | "
            f"Mean: {mean_lines_kept:.4f} | Median: {median_lines_kept:.4f}"
        )
        print(
            f"Characters: {total_chars_kept:,}/{total_chars_seen:,} ({chars_survival:.4f}% survival)"
        )
        print(
            f"Duration: {duration_kept:.4f}/{duration_seen:.4f} hours ({duration_survival:.4f}% survival)"
        )

        if subsample:
            subsample_survival = (
                (100 * total_subsampled / total_lines_seen)
                if total_lines_seen > 0
                else 0.0
            )
            print(
                f"Subsampled: {total_subsampled:,} lines ({subsampled_duration:.4f} hours, {subsample_survival:.4f}% survival)"
            )

        # Display filtering breakdown
        if config_dict.get("pipeline"):
            display_hitlist_report(dict(total_hitlist), config_dict["pipeline"])

        # Save detailed log
        subsample_stats = (total_subsampled, subsampled_duration) if subsample else None
        report = generate_processing_report(
            files_processed=len(files),
            processing_time=processing_time,
            lines_stats=(total_lines_seen, total_lines_kept),
            chars_stats=(total_chars_seen, total_chars_kept),
            duration_stats=(duration_seen, duration_kept),
            hitlist=dict(total_hitlist),
            pipeline=config_dict.get("pipeline", []),
            subsample_stats=subsample_stats,
        )

        save_processing_log(report, output_dir, config_path)

    print(f"\nResults saved to: {output_dir}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace

    Raises:
        SystemExit: If argument parsing fails
    """
    parser = argparse.ArgumentParser(
        description="Process tagged JSONL subtitle data with filtering and subsampling",
        epilog="Example: python process_tagged_data.py --input-dir /data --output-dir /output --config filters.yaml",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input JSONL.gz files",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where processed JSONL.gz files will be saved",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to YAML configuration file defining the filtering pipeline",
    )

    parser.add_argument(
        "--only-subsample",
        action="store_true",
        help="Only subsample the data without applying filters",
    )

    parser.add_argument(
        "--subsample", action="store_true", help="Subsample the data after filtering"
    )

    parser.add_argument(
        "--subsample-size",
        type=int,
        default=1000,
        help="Number of samples to keep when subsampling (default: 1000)",
    )

    parser.add_argument(
        "--num-cpus",
        type=int,
        required=False,
        help="Number of CPU cores to use (default: all available)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_arguments()

        main(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            only_subsample=args.only_subsample,
            subsample=args.subsample,
            subsample_size=args.subsample_size,
            num_cpus=args.num_cpus,
        )

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)