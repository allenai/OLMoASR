"""
Data tagging pipeline for subtitle/transcript content analysis.

This module provides a comprehensive tagging system for analyzing subtitle and transcript
content, including language detection, text quality metrics, formatting analysis,
and content validation. It processes JSONL files containing subtitle data and applies
configurable tagging pipelines to extract various content features.

Features:
- Language identification using CLD2
- Edit distance calculation between manual and machine transcripts
- Text formatting analysis (casing, punctuation, repetition)
- Content quality metrics (word count, proper capitalization)
- Parallel processing for large datasets
- Configurable tagging pipelines via YAML configuration
"""

import os
import glob
import json
import gzip
import re
from collections import defaultdict
from functools import partial
from io import StringIO
from itertools import repeat
from multiprocessing import Pool
from typing import Dict, List, Tuple, Union, Optional, Any, Literal, Callable

import yaml
from fire import Fire
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
    from whisper.normalizers import EnglishTextNormalizer  # type: ignore

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    EnglishTextNormalizer = None

try:
    import jiwer  # type: ignore

    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    jiwer = None

try:
    import webvtt  # type: ignore

    WEBVTT_AVAILABLE = True
except ImportError:
    WEBVTT_AVAILABLE = False
    webvtt = None

# Constants for character classification
LOWERCASE_LETTERS = set([chr(ord("a") + i) for i in range(26)])
UPPERCASE_LETTERS = set(_.upper() for _ in LOWERCASE_LETTERS)

# Type aliases for better readability
SubtitleContent = Union[str, Any]  # Can be subtitle content or parsed subtitle object
TagValue = Union[str, float, bool, int]
TagStats = Dict[str, Any]  # Flexible type for statistics
ProcessingResult = Tuple[int, int, float, Dict[str, Any]]
ConfigDict = Dict[str, Any]


def run_parallel_processing(
    func: Callable, argument_list: List[Any], num_processes: int
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


def load_config(config_path: str) -> ConfigDict:
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


def extract_manual_text(manual_content: str) -> str:
    """
    Extract text from manual subtitle content.

    Args:
        manual_content: Raw manual subtitle content (VTT or SRT format)

    Returns:
        Extracted plain text from subtitles

    Raises:
        ImportError: If required dependencies are not available
        ValueError: If content cannot be parsed
    """
    if not OLMOASR_AVAILABLE or TranscriptReader is None:
        raise ImportError("olmoasr.utils is required but not available")

    try:
        # Determine format based on content
        subtitle_format = "vtt" if manual_content.startswith("WEBVTT") else "srt"

        reader = TranscriptReader(
            file_path=None,
            transcript_string=manual_content,
            ext=subtitle_format,
        )

        transcript_dict, *_ = reader.read()
        manual_text = reader.extract_text(transcript_dict)

        return manual_text
    except Exception as e:
        raise ValueError(f"Failed to extract manual text: {e}")


def extract_machine_text(machine_content: str) -> str:
    """
    Extract text from machine-generated subtitle content.

    Args:
        machine_content: Raw machine subtitle content (VTT format)

    Returns:
        Extracted and cleaned plain text from machine subtitles

    Raises:
        ImportError: If required dependencies are not available
        ValueError: If content cannot be parsed
    """
    if not WEBVTT_AVAILABLE or webvtt is None:
        raise ImportError("webvtt is required but not available")

    try:
        content = webvtt.from_string(machine_content)

        if not content:
            return ""

        # Handle duplicate first caption
        start_index = _get_start_index_for_machine_content(content)

        # Filter and clean captions
        cleaned_captions = _filter_machine_captions(content, start_index)

        # Extract text from cleaned captions
        machine_text = " ".join([caption.text for caption in cleaned_captions])

        return machine_text
    except Exception as e:
        raise ValueError(f"Failed to extract machine text: {e}")


def _get_start_index_for_machine_content(content: Any) -> int:
    """
    Determine the starting index for machine content processing.

    Args:
        content: Parsed WebVTT content

    Returns:
        Starting index for processing captions
    """
    if len(content) <= 1:
        return 0

    # Skip first caption if it's a duplicate of the second
    if content[0].text == content[1].text:
        return 2

    return 0


def _filter_machine_captions(content: Any, start_index: int) -> List[Any]:
    """
    Filter machine captions to remove multi-line captions except for the last one.

    Args:
        content: Parsed WebVTT content
        start_index: Index to start filtering from

    Returns:
        List of filtered captions
    """
    filtered_captions = []

    for i in range(start_index, len(content)):
        caption = content[i]

        if "\n" not in caption.text:
            filtered_captions.append(caption)
        elif "\n" in caption.text and i == len(content) - 1:
            # For the last caption, take only the last line
            caption.text = caption.text.split("\n")[-1]
            filtered_captions.append(caption)

    return filtered_captions


def parse_subtitle_content(content: str, subtitle_filename: str) -> Any:
    """
    Parse subtitle content into an iterable format.

    Args:
        content: Raw subtitle content
        subtitle_filename: Filename to determine format (extension-based)

    Returns:
        Parsed subtitle content as an iterable

    Raises:
        ImportError: If required dependencies are not available
        ValueError: If unsupported file format or parsing fails
    """
    file_extension = os.path.splitext(subtitle_filename)[-1].lower()

    try:
        if file_extension == ".vtt":
            if not WEBVTT_AVAILABLE or webvtt is None:
                raise ImportError("webvtt is required but not available")
            return webvtt.from_string(content)
        else:
            raise ValueError(f"Unsupported subtitle format: {file_extension}")
    except Exception as e:
        raise ValueError(f"Failed to parse subtitle content: {e}")


def calculate_edit_distance(
    content_dict: Dict[str, Any], normalizer: Any
) -> Tuple[float, TagStats]:
    """
    Calculate edit distance between manual and machine text.

    Args:
        content_dict: Dictionary containing manual and machine text
        normalizer: Text normalizer for preprocessing

    Returns:
        Tuple of (edit_distance, statistics_dict)

    Raises:
        ImportError: If required dependencies are not available
    """
    if not JIWER_AVAILABLE or jiwer is None:
        raise ImportError("jiwer is required for edit distance calculation")

    stats = {"count_0": 0, "count_1": 0, "count_gt_1": 0, "count_lt_1": 0}

    manual_text = content_dict["man_text"].strip()
    machine_text = content_dict["mach_text"].strip()

    # Normalize texts with error handling
    normalized_manual = _safe_normalize_text(manual_text, normalizer)
    normalized_machine = _safe_normalize_text(machine_text, normalizer)

    # Calculate edit distance based on available text
    edit_distance = _compute_edit_distance(
        manual_text, machine_text, normalized_manual, normalized_machine
    )

    # Update statistics
    _update_edit_distance_stats(edit_distance, stats)

    return edit_distance, stats


def _safe_normalize_text(text: str, normalizer: Any) -> str:
    """
    Safely normalize text with error handling.

    Args:
        text: Text to normalize
        normalizer: Text normalizer

    Returns:
        Normalized text or original text if normalization fails
    """
    try:
        return normalizer(text).strip()
    except Exception:
        return text


def _compute_edit_distance(
    manual_text: str, machine_text: str, norm_manual: str, norm_machine: str
) -> float:
    """
    Compute edit distance between texts with various fallback strategies.

    Args:
        manual_text: Original manual text
        machine_text: Original machine text
        norm_manual: Normalized manual text
        norm_machine: Normalized machine text

    Returns:
        Edit distance as float
    """
    if not JIWER_AVAILABLE or jiwer is None:
        return 0.0

    if norm_manual != "":
        return jiwer.wer(norm_manual, norm_machine)
    elif manual_text == "":
        if norm_machine != "":
            return jiwer.wer(norm_machine, manual_text)
        elif machine_text != "":
            return jiwer.wer(machine_text, manual_text)
        else:
            return 0.0
    else:
        return jiwer.wer(manual_text, norm_machine)


def _update_edit_distance_stats(edit_distance: float, stats: TagStats) -> None:
    """
    Update edit distance statistics.

    Args:
        edit_distance: Calculated edit distance
        stats: Statistics dictionary to update
    """
    if edit_distance == 0.0:
        stats["count_0"] += 1
    elif edit_distance == 1.0:
        stats["count_1"] += 1
    elif edit_distance > 1.0:
        stats["count_gt_1"] += 1
    else:  # 0.0 < edit_distance < 1.0
        stats["count_lt_1"] += 1


def identify_text_language(content_dict: Dict[str, Any]) -> Tuple[str, TagStats]:
    """
    Identify the language of manual text content.

    Args:
        content_dict: Dictionary containing manual text and length

    Returns:
        Tuple of (language_code, statistics_dict)

    Raises:
        ImportError: If required dependencies are not available
    """
    if not CLD2_AVAILABLE or cld2 is None:
        raise ImportError("pycld2 is required for language identification")

    stats = {
        "text_en_count": 0,
        "non_text_en_count": 0,
        "text_en_dur": 0,
        "non_text_en_dur": 0,
    }

    manual_text = content_dict["man_text"]
    content_length = content_dict["length"]

    try:
        # Detect language using CLD2
        *_, details = cld2.detect(manual_text)
        language_id = details[0][1]
    except Exception:
        # Fallback to English if detection fails
        language_id = "en"

    # Update statistics based on language
    if language_id == "en":
        stats["text_en_count"] += 1
        stats["text_en_dur"] += content_length
    else:
        stats["non_text_en_dur"] += content_length

    stats["non_text_en_count"] = 1 - stats["text_en_count"]

    return language_id, stats


def analyze_text_casing(content_dict: Dict[str, Any]) -> Tuple[str, TagStats]:
    """
    Analyze the casing pattern of subtitle content.

    Args:
        content_dict: Dictionary containing content iterator and length

    Returns:
        Tuple of (casing_type, statistics_dict)
    """
    content_iter = content_dict["content_iter"]
    content_length = content_dict["length"]

    stats = {
        "count_upper": 0,
        "count_lower": 0,
        "count_mixed": 0,
        "dur_upper": 0,
        "dur_lower": 0,
        "dur_mixed": 0,
    }

    # Count casing types across all captions
    casing_counts = {"upper": 0, "lower": 0, "mixed": 0}

    for caption in content_iter:
        casing_type = _classify_caption_casing(caption.text)
        casing_counts[casing_type] += 1

    # Determine overall casing with adjustment rules
    final_casing = _determine_final_casing(casing_counts)

    # Update statistics
    _update_casing_stats(final_casing, content_length, stats)

    return final_casing, stats


def _classify_caption_casing(text: str) -> str:
    """
    Classify the casing of a single caption.

    Args:
        text: Caption text to classify

    Returns:
        Casing type: 'upper', 'lower', or 'mixed'
    """
    if not text.strip():
        return "mixed"

    char_set = set(text)
    has_upper = bool(UPPERCASE_LETTERS.intersection(char_set))
    has_lower = bool(LOWERCASE_LETTERS.intersection(char_set))

    if has_upper and has_lower:
        return "mixed"
    elif has_upper:
        return "upper"
    else:
        return "lower"


def _determine_final_casing(casing_counts: Dict[str, int]) -> str:
    """
    Determine final casing type based on counts with adjustment rules.

    Args:
        casing_counts: Dictionary of casing type counts

    Returns:
        Final casing determination
    """
    max_count = max(casing_counts.values())
    max_keys = [key for key, count in casing_counts.items() if count == max_count]

    if len(max_keys) == 1:
        dominant_type = max_keys[0]

        # Apply adjustment rules
        if dominant_type == "lower" and casing_counts["mixed"] / max_count > 0.6:
            return "mixed"
        elif dominant_type == "mixed" and casing_counts["upper"] / max_count > 0.6:
            return "upper"
        else:
            return dominant_type
    else:
        # Multiple tied types - prefer mixed if present
        return "mixed" if "mixed" in max_keys else max_keys[0]


def _update_casing_stats(casing: str, content_length: float, stats: TagStats) -> None:
    """
    Update casing statistics.

    Args:
        casing: Final casing determination
        content_length: Length of content
        stats: Statistics dictionary to update
    """
    if casing == "upper":
        stats["count_upper"] += 1
        stats["dur_upper"] += content_length
    elif casing == "lower":
        stats["count_lower"] += 1
        stats["dur_lower"] += content_length
    elif casing == "mixed":
        stats["count_mixed"] += 1
        stats["dur_mixed"] += content_length


def check_comma_period_presence(content_dict: Dict[str, Any]) -> Tuple[bool, TagStats]:
    """
    Check if content contains both commas and periods.

    Args:
        content_dict: Dictionary containing content iterator and length

    Returns:
        Tuple of (has_both_punctuation, statistics_dict)
    """
    content_iter = content_dict["content_iter"]
    content_length = content_dict["length"]

    stats = {"count": 0, "dur": 0}

    has_period = has_comma = False

    for caption in content_iter:
        if not has_period and "." in caption.text:
            has_period = True
        if not has_comma and "," in caption.text:
            has_comma = True

        if has_period and has_comma:
            stats["count"] += 1
            stats["dur"] += content_length
            return True, stats

    return False, stats


def detect_repeating_lines(content_dict: Dict[str, Any]) -> Tuple[bool, TagStats]:
    """
    Detect if content has repeating lines.

    Args:
        content_dict: Dictionary containing content iterator and length

    Returns:
        Tuple of (has_repeating_lines, statistics_dict)
    """
    content_iter = content_dict["content_iter"]
    content_length = content_dict["length"]

    stats = {"count": 0, "dur": 0}

    text_history = []

    for caption in content_iter:
        current_text = caption.text

        # Check if current text contains any previous text
        if text_history and _has_repeated_content(current_text, text_history):
            stats["count"] += 1
            stats["dur"] += content_length
            return True, stats

        text_history.append(current_text)

    return False, stats


def _has_repeated_content(current_text: str, text_history: List[str]) -> bool:
    """
    Check if current text contains repeated content from history.

    Args:
        current_text: Current caption text
        text_history: List of previous caption texts

    Returns:
        True if repetition is detected
    """
    if not text_history:
        return False

    previous_text = text_history[-1]

    # Check if previous text is contained in current text
    if previous_text in current_text:
        # Only consider it repetition if both texts have multiple words
        current_words = len(current_text.strip().split())
        previous_words = len(previous_text.strip().split())

        return current_words > 1 and previous_words > 1

    return False


def check_proper_capitalization(content_dict: Dict[str, Any]) -> Tuple[bool, TagStats]:
    """
    Check if content has proper capitalization after punctuation.

    Args:
        content_dict: Dictionary containing content iterator and length

    Returns:
        Tuple of (has_proper_capitalization, statistics_dict)
    """
    content_iter = content_dict["content_iter"]
    content_length = content_dict["length"]

    stats = {"count": 0, "dur": 0}

    # Pattern for sentence-ending punctuation at line end
    punctuation_pattern = r"[.!?](?:\s*)$"

    for i, caption in enumerate(content_iter):
        if i == 0:
            continue

        previous_caption = content_iter[i - 1]

        # Check if previous caption ends with punctuation
        if re.search(punctuation_pattern, previous_caption.text):
            current_text = caption.text.strip()

            # Check if current caption starts with proper capitalization
            if (
                current_text
                and current_text[0].isalpha()
                and not current_text[0].isupper()
            ):
                return False, stats

    # All capitalization checks passed
    stats["count"] += 1
    stats["dur"] += content_length
    return True, stats


def count_words(content_dict: Dict[str, Any]) -> Tuple[int, None]:
    """
    Count the total number of words in content.

    Args:
        content_dict: Dictionary containing content iterator

    Returns:
        Tuple of (word_count, None) - no statistics for this metric
    """
    content_iter = content_dict["content_iter"]

    total_words = 0
    for caption in content_iter:
        words = caption.text.strip().split()
        total_words += len(words)

    return total_words, None


# Registry of all available tagging functions
TAGGING_FUNCTIONS = {
    "has_comma_period": check_comma_period_presence,
    "casing": analyze_text_casing,
    "repeating_lines": detect_repeating_lines,
    "edit_dist": calculate_edit_distance,
    "text_lang": identify_text_language,
    "has_proper_cap_after_punct_line": check_proper_capitalization,
    "num_words": count_words,
}


def process_jsonl_file(
    jsonl_path: str,
    config_dict: ConfigDict,
    output_dir: str,
    append_to_existing: bool = False,
    segment_level: bool = False,
) -> ProcessingResult:
    """
    Process a single JSONL file with the configured tagging pipeline.

    Args:
        jsonl_path: Path to the input JSONL.gz file
        config_dict: Configuration dictionary with tagging pipeline
        output_dir: Directory to save processed results
        append_to_existing: Whether to append tags to existing data
        segment_level: Whether to process at segment level

    Returns:
        Tuple of (lines_processed, characters_processed, duration_processed, statistics)

    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
        ValueError: If processing fails
    """
    # Setup output paths
    base_filename = os.path.basename(jsonl_path)
    output_file = os.path.join(output_dir, base_filename)
    stats_file = os.path.join(
        output_dir, base_filename.replace(".jsonl.gz", "_stats.json")
    )

    # Load and validate input data
    try:
        with gzip.open(jsonl_path, "rt", encoding="utf-8") as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]
    except Exception as e:
        raise ValueError(f"Failed to load JSONL file {jsonl_path}: {e}")

    lines_processed = len(lines)
    characters_processed = 0
    duration_processed = 0.0

    # Check if already processed
    if os.path.exists(stats_file):
        # Load existing statistics
        with open(stats_file, "r", encoding="utf-8") as f:
            existing_stats = json.load(f)

        # Calculate metrics from existing data
        characters_processed = sum(len(line.get("content", "")) for line in lines)
        duration_processed = sum(line.get("length", 0) for line in lines)

        return lines_processed, characters_processed, duration_processed, existing_stats

    # Process lines with tagging pipeline
    output_lines = []
    file_statistics = []
    tag_accumulator = defaultdict(list) if append_to_existing else None

    for line in lines:
        # Prepare content dictionary
        content_dict = _prepare_content_dict(line, segment_level)

        # Update processing metrics
        if not segment_level:
            characters_processed += len(line.get("content", ""))
            duration_processed += line.get("length", 0)

        # Apply tagging pipeline
        tags, stats = apply_tagging_pipeline(content_dict, config_dict)

        # Store results
        if append_to_existing:
            for tag, value in tags.items():
                tag_accumulator[tag].append(value)
        else:
            line.update(tags)
            output_lines.append(line)

        if stats:
            file_statistics.append(stats)

    # Save results
    _save_processing_results(
        output_file,
        stats_file,
        output_lines,
        file_statistics,
        tag_accumulator,
        lines,
        append_to_existing,
        segment_level,
    )

    # Calculate cumulative statistics
    cumulative_stats = aggregate_statistics(file_statistics) if file_statistics else {}

    return lines_processed, characters_processed, duration_processed, cumulative_stats


def _prepare_content_dict(line: Dict[str, Any], segment_level: bool) -> Dict[str, Any]:
    """
    Prepare content dictionary for processing.

    Args:
        line: Input line from JSONL file
        segment_level: Whether processing at segment level

    Returns:
        Content dictionary for tagging functions
    """
    if segment_level:
        return {
            "content_iter": parse_subtitle_content(
                line["seg_content"], line["subtitle_file"]
            )
        }
    else:
        return {
            "content_iter": parse_subtitle_content(
                line["content"], line["subtitle_file"]
            ),
            "man_text": extract_manual_text(line["content"]),
            "mach_text": (
                extract_machine_text(line["mach_content"])
                if line.get("mach_content", "")
                else ""
            ),
            "length": line["length"],
        }


def _save_processing_results(
    output_file: str,
    stats_file: str,
    output_lines: List[Dict[str, Any]],
    file_statistics: List[Dict[str, Any]],
    tag_accumulator: Optional[Dict[str, List[TagValue]]],
    original_lines: List[Dict[str, Any]],
    append_to_existing: bool,
    segment_level: bool,
) -> None:
    """
    Save processing results to files.

    Args:
        output_file: Path to output file
        stats_file: Path to statistics file
        output_lines: Processed lines with tags
        file_statistics: Accumulated statistics
        tag_accumulator: Tags accumulated for appending
        original_lines: Original input lines
        append_to_existing: Whether appending to existing data
        segment_level: Whether processing at segment level
    """
    # Save main output
    if append_to_existing and tag_accumulator:
        # Create ID-to-tags mapping
        id_key = "seg_id" if segment_level else "id"
        id_to_tags = {
            line[id_key]: {
                tag: tag_accumulator[tag][i] for tag in tag_accumulator.keys()
            }
            for i, line in enumerate(original_lines)
        }

        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            json.dump(id_to_tags, f, indent=2)
    else:
        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            for line in output_lines:
                f.write(json.dumps(line) + "\n")

    # Save statistics
    if file_statistics:
        cumulative_stats = aggregate_statistics(file_statistics)
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(cumulative_stats, f, indent=2)


def apply_tagging_pipeline(
    content_dict: Dict[str, Any], config: ConfigDict
) -> Tuple[Dict[str, TagValue], Dict[str, TagStats]]:
    """
    Apply the configured tagging pipeline to content.

    Args:
        content_dict: Content dictionary with data to tag
        config: Configuration dictionary with pipeline specification

    Returns:
        Tuple of (tags_dict, statistics_dict)

    Raises:
        KeyError: If unknown tag function is specified
        ValueError: If tag function fails
    """
    tags = {}
    statistics = {}

    for tag_config in config.get("pipeline", []):
        tag_name = tag_config["tag"]

        if tag_name not in TAGGING_FUNCTIONS:
            raise KeyError(f"Unknown tagging function: {tag_name}")

        tag_function = TAGGING_FUNCTIONS[tag_name]

        # Extract function arguments
        kwargs = {k: v for k, v in tag_config.items() if k != "tag"}

        try:
            # Apply special handling for edit distance
            if tag_name == "edit_dist":
                if not WHISPER_AVAILABLE or EnglishTextNormalizer is None:
                    raise ImportError(
                        "whisper.normalizers is required for edit distance"
                    )
                normalizer = EnglishTextNormalizer()
                tag_value, tag_stats = tag_function(content_dict, normalizer)
            else:
                tag_value, tag_stats = tag_function(content_dict, **kwargs)

            tags[tag_name] = tag_value
            if tag_stats is not None:
                statistics[tag_name] = tag_stats

        except Exception as e:
            raise ValueError(f"Failed to apply tag function {tag_name}: {e}")

    return tags, statistics


def aggregate_statistics(
    statistics_list: List[Dict[str, TagStats]]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate statistics from multiple processing results.

    Args:
        statistics_list: List of statistics dictionaries

    Returns:
        Aggregated statistics dictionary
    """
    if not statistics_list:
        return {}

    # Get all tag names
    all_tags = set()
    for stats in statistics_list:
        all_tags.update(stats.keys())

    aggregated = {}

    for tag in all_tags:
        tag_stats = [stats.get(tag, {}) for stats in statistics_list]

        # Get all stat keys for this tag
        all_stat_keys = set()
        for stat in tag_stats:
            all_stat_keys.update(stat.keys())

        aggregated_tag_stats = {}

        for stat_key in all_stat_keys:
            values = [stat.get(stat_key, 0) for stat in tag_stats]

            if "dur" in stat_key:
                # Sum duration-based statistics
                aggregated_tag_stats[stat_key] = sum(values)
            elif "count" in stat_key:
                # Average count-based statistics
                avg_key = stat_key.replace("count", "avg")
                aggregated_tag_stats[avg_key] = sum(values) / len(values)
            else:
                # Sum other statistics
                aggregated_tag_stats[stat_key] = sum(values)

        aggregated[tag] = aggregated_tag_stats

    return aggregated


def generate_summary_report(
    output_dir: str,
    config_path: str,
    processing_results: List[ProcessingResult],
    cumulative_stats: Dict[str, Dict[str, float]],
) -> None:
    """
    Generate a comprehensive summary report.

    Args:
        output_dir: Directory to save the report
        config_path: Path to configuration file
        processing_results: List of processing results
        cumulative_stats: Aggregated statistics
    """
    # Aggregate processing metrics
    if not processing_results:
        return
    total_lines, total_chars, total_duration, *_ = zip(*processing_results)

    config_basename = os.path.basename(config_path).replace(".yaml", "")
    report_path = os.path.join(output_dir, f"{config_basename}_cumulative_stats.log")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("DATA TAGGING PIPELINE SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Configuration: {config_path}\n")
        f.write(f"Output Directory: {output_dir}\n\n")

        f.write("PROCESSING SUMMARY:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total files processed: {len(processing_results)}\n")
        f.write(f"Total lines/videos: {sum(total_lines)}\n")
        f.write(f"Total characters: {sum(total_chars):,}\n")
        f.write(f"Total duration: {sum(total_duration) / 3600:.2f} hours\n\n")

        f.write("TAGGING RESULTS:\n")
        f.write("-" * 30 + "\n")

        for tag, tag_stats in cumulative_stats.items():
            f.write(f"{tag.upper()}:\n")
            for stat_name, value in tag_stats.items():
                if "dur" in stat_name:
                    f.write(f"  {stat_name}: {value / 3600:.2f} hours\n")
                else:
                    f.write(f"  {stat_name}: {value:.4f}\n")
            f.write("\n")


def main(
    config_path: str,
    input_dir: str,
    output_dir: str,
    num_cpus: Optional[int] = None,
    append_to_existing: bool = False,
    segment_level: bool = False,
) -> None:
    """
    Main entry point for the data tagging pipeline.

    Args:
        config_path: Path to YAML configuration file
        input_dir: Directory containing input .jsonl.gz files
        output_dir: Directory to save processed results
        num_cpus: Number of CPU cores to use (default: all available)
        append_to_existing: Whether to append tags to existing data
        segment_level: Whether to process at segment level

    Raises:
        FileNotFoundError: If input directory or config file doesn't exist
        ValueError: If invalid parameters are provided
    """
    # Validate inputs
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Set up processing parameters
    if num_cpus is None:
        num_cpus = os.cpu_count() or 1

    os.makedirs(output_dir, exist_ok=True)

    # Find input files
    input_files = glob.glob(os.path.join(input_dir, "*.jsonl.gz"))
    if not input_files:
        raise FileNotFoundError(f"No .jsonl.gz files found in {input_dir}")

    # Load configuration
    config_dict = load_config(config_path)
    print(f"Configuration loaded: {config_dict}")
    print(f"Processing {len(input_files)} files with {num_cpus} CPU cores")

    # Create partial function for parallel processing
    process_function = partial(
        process_jsonl_file,
        config_dict=config_dict,
        output_dir=output_dir,
        append_to_existing=append_to_existing,
        segment_level=segment_level,
    )

    # Process files in parallel
    processing_results = run_parallel_processing(
        process_function, input_files, num_cpus
    )

    # Aggregate final statistics
    all_stats = [result[3] for result in processing_results if result[3]]
    cumulative_stats = aggregate_statistics(all_stats)

    # Generate summary report
    generate_summary_report(
        output_dir, config_path, processing_results, cumulative_stats
    )

    print(f"Processing complete. Results saved to {output_dir}")


if __name__ == "__main__":
    Fire(main)