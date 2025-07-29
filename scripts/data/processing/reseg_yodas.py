"""
YODAS Audio Resegmentation Script

This script processes YODAS (YouTube-based Open Dataset for Audio Segmentation) 
dataset by resegmenting audio files based on duration constraints and context 
length limitations for speech recognition models.

The script performs the following operations:
1. Loads YODAS dataset from Arrow files
2. Resegments audio based on 30-second duration limits
3. Checks context length constraints for tokenization
4. Generates new concatenated audio segments
5. Exports processed segments to JSONL format

Features:
- Parallel processing for performance
- Context length validation for speech models
- Audio concatenation with duration limits
- Timestamp extraction and processing
- Error handling for malformed data

Usage:
    python reseg_yodas.py --input_dir="/path/to/yodas" --audio_output_dir="/path/to/audio" --text_output_dir="/path/to/text"

Dependencies:
    - datasets: For loading YODAS dataset
    - soundfile: For audio file operations
    - whisper: For tokenization and context length checking
    - numpy: For audio array operations
    - tqdm: For progress tracking
    - fire: For command-line interface
"""

import os
import re
import json
import glob
import multiprocessing
from typing import Optional, List, Tuple, Dict, Any, Union
from itertools import repeat

import numpy as np
import soundfile as sf
from datasets import Dataset
from tqdm import tqdm
from whisper.tokenizer import get_tokenizer
from fire import Fire

# Type aliases for better readability
FilePath = str
UtteranceID = str
AudioArray = np.ndarray
Timestamp = float
TimestampPair = Tuple[Timestamp, Timestamp]
SegmentData = Dict[str, Any]
ContextResult = Dict[str, Union[bool, int]]

# Global variable for multiprocessing
shared_ds: Optional[Dataset] = None


def init_worker(dataset: Dataset) -> None:
    """
    Initialize worker process with shared dataset.

    Args:
        dataset: The dataset to share across worker processes
    """
    global shared_ds
    shared_ds = dataset


def extract_id(utt_id: UtteranceID) -> str:
    """
    Extract base ID from utterance ID by removing segment numbers.

    Args:
        utt_id: Full utterance ID (e.g., "speaker-audio-12345-start-end")

    Returns:
        Base ID without segment numbers

    Raises:
        ValueError: If utterance ID format is invalid
    """
    match = re.search(r"^(.*?)-\d{5}(?!\d)", utt_id)

    if match is None:
        raise ValueError(f"Invalid utterance ID format: {utt_id}")

    return match.group(1)


def check_over_ctx_len(
    timestamps: List[TimestampPair],
    transcript_list: List[str],
    language: Optional[str] = None,
    last_seg: bool = False,
) -> Tuple[bool, Optional[ContextResult]]:
    """
    Check if transcript text exceeds model context length.

    Validates whether the transcript tokens fit within the model's context window
    for both timestamp and non-timestamp modes.

    Args:
        timestamps: List of (start, end) timestamp pairs
        transcript_list: List of transcript text segments
        language: Language code for tokenizer (None for English)
        last_seg: Whether this is the last segment in the sequence

    Returns:
        Tuple of (over_limit, context_info) where:
        - over_limit: True if exceeds context length in both modes
        - context_info: Dict with mode availability and token counts, or None if over limit

    Raises:
        RuntimeError: If tokenizer fails to process text
    """
    try:
        # Initialize tokenizer based on language
        if language is None:
            tokenizer = get_tokenizer(multilingual=False)
        else:
            tokenizer = get_tokenizer(language=language, multilingual=True)

        # Tokenize transcript text
        text_tokens = [tokenizer.encode(" " + text.strip()) for text in transcript_list]

        # Calculate token counts for different modes
        num_timestamp_tokens = (
            (len(timestamps) * 2) + 1 if not last_seg else (len(timestamps) * 2)
        )  # Add next_start timestamp (+1) when not last segment

        num_text_tokens = sum(len(token_group) for token_group in text_tokens)
        num_tokens_ts_mode = num_timestamp_tokens + num_text_tokens + 2  # sot + eot
        num_tokens_no_ts_mode = num_text_tokens + 3  # sot + notimestamps + eot

        # Check context length constraints (448 tokens max)
        if num_tokens_ts_mode > 448 and num_tokens_no_ts_mode > 448:
            return True, None
        elif num_tokens_ts_mode > 448 and num_tokens_no_ts_mode <= 448:
            return False, {
                "ts_mode": False,
                "no_ts_mode": True,
                "num_tokens_no_ts_mode": num_tokens_no_ts_mode,
                "num_tokens_ts_mode": num_tokens_ts_mode,
            }
        elif num_tokens_ts_mode <= 448 and num_tokens_no_ts_mode > 448:
            return False, {
                "ts_mode": True,
                "no_ts_mode": False,
                "num_tokens_no_ts_mode": num_tokens_no_ts_mode,
                "num_tokens_ts_mode": num_tokens_ts_mode,
            }
        else:
            return False, {
                "ts_mode": True,
                "no_ts_mode": True,
                "num_tokens_no_ts_mode": num_tokens_no_ts_mode,
                "num_tokens_ts_mode": num_tokens_ts_mode,
            }
    except (RuntimeError, Exception) as e:
        print(f"Error in tokenization: {e}")
        return True, None


def extract_ts(
    utt_id: UtteranceID,
    extract_type: Optional[str] = None,
    global_start: Optional[Timestamp] = None,
) -> Union[Timestamp, TimestampPair]:
    """
    Extract timestamp information from utterance ID.

    Args:
        utt_id: Utterance ID containing timestamp information
        extract_type: Type of timestamp to extract ("start", "end", or None for both)
        global_start: Global start time to subtract for relative timestamps

    Returns:
        Single timestamp (float) or timestamp pair (tuple) depending on extract_type

    Raises:
        ValueError: If utterance ID format is invalid
    """
    try:
        # Extract timestamp parts from utterance ID
        parts = utt_id.split("-")
        if len(parts) < 2:
            raise ValueError(f"Invalid utterance ID format: {utt_id}")

        raw_start, raw_end = parts[-2], parts[-1]

        # Parse timestamps (format: integer + 2-digit decimal)
        start = int(raw_start[:-2]) + float(f"0.{raw_start[-2:]}")
        end = int(raw_end[:-2]) + float(f"0.{raw_end[-2:]}")

        # Apply global start offset if provided
        if global_start is not None:
            start -= global_start
            end -= global_start

        # Return based on requested type
        if extract_type == "start":
            return start
        elif extract_type == "end":
            return end
        else:
            return (start, end)

    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to extract timestamp from {utt_id}: {e}")


def extract_ts_list(utt_id_list: List[UtteranceID]) -> List[TimestampPair]:
    """
    Extract timestamp pairs for a list of utterance IDs.

    Args:
        utt_id_list: List of utterance IDs

    Returns:
        List of (start, end) timestamp pairs relative to the first utterance

    Raises:
        ValueError: If any utterance ID is invalid
    """
    if not utt_id_list:
        return []

    # Get global start time from first utterance
    global_start = extract_ts(utt_id_list[0], "start")

    # Extract relative timestamps for all utterances
    timestamp_pairs = []
    for utt_id in utt_id_list:
        ts_pair = extract_ts(utt_id, None, global_start)
        if isinstance(ts_pair, tuple):
            timestamp_pairs.append(ts_pair)
        else:
            raise ValueError(f"Expected timestamp pair, got {type(ts_pair)}")

    return timestamp_pairs


def reseg_data(dataset: Dataset) -> List[Tuple[List[int], Union[str, int]]]:
    """
    Resegment dataset based on duration constraints.

    Groups consecutive utterances from the same speaker/source into segments
    that don't exceed 30 seconds duration.

    Args:
        dataset: Dataset containing utterance data

    Returns:
        List of tuples containing (segment_indices, segment_info)

    Raises:
        ValueError: If dataset format is invalid
    """
    if not dataset or dataset.num_rows == 0:
        return []

    # Sort utterances by ID for chronological processing
    utt_id_list = sorted(zip(dataset["utt_id"], dataset["id"]), key=lambda x: x[0])
    segments = []
    seg_count = 0
    a, b = 0, 0

    with tqdm(total=dataset.num_rows, desc="Resegmenting data") as pbar:
        while b < len(utt_id_list):
            try:
                # Calculate duration for current segment
                start_time = extract_ts(utt_id_list[a][0], "start")
                end_time = extract_ts(utt_id_list[b][0], "end")

                if isinstance(start_time, tuple) or isinstance(end_time, tuple):
                    raise ValueError("Expected single timestamp values")

                dur = end_time - start_time

                # Check if duration is within 30-second limit
                if dur <= 30.00:
                    # Check if utterances are from the same source
                    if b < len(utt_id_list) - 1 and extract_id(
                        utt_id_list[b][0]
                    ) == extract_id(utt_id_list[a][0]):
                        b += 1
                    else:
                        # Create segment for utterances a to b
                        seg_count += 1
                        max_30_segs = [tpl[-1] for tpl in utt_id_list[a : b + 1]]
                        segments.append((max_30_segs, seg_count - 1))
                        a = b + 1
                        b = a
                        seg_count = 0
                else:
                    # Duration exceeds limit, create segment
                    seg_count += 1
                    if a == b:
                        # Single utterance segment
                        max_30_segs = [utt_id_list[a][-1]]
                        segments.append((max_30_segs, seg_count - 1))
                        a += 1
                        b = a
                    else:
                        # Multi-utterance segment up to b-1
                        max_30_segs = [tpl[-1] for tpl in utt_id_list[a:b]]
                        segments.append((max_30_segs, seg_count - 1))
                        a = b
                    seg_count = 0

                pbar.update(1)

            except (ValueError, IndexError) as e:
                print(f"Error processing utterance {utt_id_list[b][0]}: {e}")
                b += 1
                continue

        # Handle any remaining utterances
        if a < len(utt_id_list):
            seg_count += 1
            max_30_segs = [tpl[-1] for tpl in utt_id_list[a:]]
            segments.append((max_30_segs, "last_seg"))

    return segments


def generate_new_segment(
    segment_info: Tuple[List[int], Union[str, int]], output_dir: FilePath
) -> Optional[SegmentData]:
    """
    Generate a new audio segment from combined utterances.

    Args:
        segment_info: Tuple of (segment_indices, segment_id)
        output_dir: Directory to save generated audio files

    Returns:
        Dictionary containing segment metadata, or None if generation fails
    """
    global shared_ds

    if shared_ds is None:
        print("Error: Shared dataset not initialized")
        return None

    try:
        new_seg_idxs, seg_id = segment_info

        # Get combined segment data
        combined_seg = shared_ds[new_seg_idxs]

        # Determine if this is the last segment
        is_last_seg = seg_id == "last_seg"
        idx = seg_id if isinstance(seg_id, int) else len(new_seg_idxs) - 1

        # Extract utterance information
        utt_id_list = combined_seg["utt_id"]
        if not utt_id_list:
            return None

        # Generate new utterance ID
        base_parts = utt_id_list[0].split("-")[:-3]
        start_ts = utt_id_list[0].split("-")[-2]
        end_ts = utt_id_list[-1].split("-")[-1]
        new_utt_id = f"{'-'.join(base_parts)}-{idx:05d}-{start_ts}-{end_ts}"

        # Generate audio file path
        new_audio_path = os.path.join(output_dir, f"{new_utt_id}.wav")

        # Concatenate audio arrays
        audio_dicts = combined_seg["audio"]
        concatenated_audio = np.concatenate([d["array"] for d in audio_dicts])

        # Limit audio to 30 seconds (480,000 samples at 16kHz)
        max_samples = 480000
        if concatenated_audio.shape[0] > max_samples:
            concatenated_audio = concatenated_audio[:max_samples]

        # Save audio file
        sf.write(new_audio_path, concatenated_audio, 16000)

        # Prepare text and timestamp data
        text_list = combined_seg["text"]
        if len(text_list) > 1:
            text_list = text_list[:-1]  # Remove last element if multiple

        utt_list_for_ts = utt_id_list[:-1] if len(utt_id_list) > 1 else utt_id_list
        ts_list = extract_ts_list(utt_list_for_ts)

        # Check context length constraints
        over_ctx_len, ctx_result = check_over_ctx_len(
            ts_list, text_list, None, is_last_seg
        )

        if over_ctx_len:
            # Clean up audio file if context length exceeded
            if os.path.exists(new_audio_path):
                os.remove(new_audio_path)
            return None

        # Return segment metadata
        return {
            "utt_id": new_utt_id,
            "audio": new_audio_path,
            "text": text_list,
            "ts": ts_list,
            "dur": concatenated_audio.shape[0] / 16000,
            "ts_mode": ctx_result["ts_mode"] if ctx_result else False,
            "no_ts_mode": ctx_result["no_ts_mode"] if ctx_result else False,
        }

    except Exception as e:
        print(f"Error generating segment: {e}")
        return None


def parallel_generate_segment(args: Tuple[Any, FilePath]) -> Optional[SegmentData]:
    """
    Wrapper function for parallel segment generation.

    Args:
        args: Tuple of (segment_info, output_dir)

    Returns:
        Generated segment data or None if failed
    """
    return generate_new_segment(*args)


def process_arrow_file(
    arrow_file: FilePath,
    audio_output_dir: FilePath,
    text_output_dir: FilePath,
    shard_index: int,
) -> None:
    """
    Process a single Arrow file from the YODAS dataset.

    Args:
        arrow_file: Path to the Arrow file
        audio_output_dir: Directory for output audio files
        text_output_dir: Directory for output text files
        shard_index: Index for naming output files
    """
    output_file = os.path.join(text_output_dir, f"shard_seg_{shard_index:05d}.jsonl")

    if os.path.exists(output_file):
        print(f"shard_seg_{shard_index:05d}.jsonl already exists, skipping")
        return

    try:
        # Load dataset from Arrow file
        dataset = Dataset.from_file(arrow_file)
        print(f"Processing {arrow_file} with {dataset.num_rows} rows")

        # Resegment the data
        print("Resegmenting data...")
        segments = reseg_data(dataset)
        print(f"Generated {len(segments)} segments")

        # Process segments in parallel
        print("Generating new segments...")
        with multiprocessing.Pool(initializer=init_worker, initargs=(dataset,)) as pool:
            new_segments = list(
                tqdm(
                    pool.imap_unordered(
                        parallel_generate_segment,
                        zip(segments, repeat(audio_output_dir)),
                    ),
                    total=len(segments),
                    desc="Processing segments",
                )
            )

        # Filter out failed segments
        valid_segments = [seg for seg in new_segments if seg is not None]
        print(f"Successfully generated {len(valid_segments)} valid segments")

        # Write results to JSONL file
        print(f"Writing segments to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            for item in valid_segments:
                f.write(json.dumps(item) + "\n")

        print(f"Completed processing {arrow_file}")

    except Exception as e:
        print(f"Error processing {arrow_file}: {e}")


def main(
    input_dir: FilePath, audio_output_dir: FilePath, text_output_dir: FilePath
) -> None:
    """
    Main function to process YODAS dataset resegmentation.

    Args:
        input_dir: Directory containing YODAS Arrow files
        audio_output_dir: Directory for output audio files
        text_output_dir: Directory for output text files

    Raises:
        ValueError: If input directory doesn't exist or contains no Arrow files
        IOError: If output directories cannot be created
    """
    # Validate input directory
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Create output directories
    try:
        os.makedirs(audio_output_dir, exist_ok=True)
        os.makedirs(text_output_dir, exist_ok=True)
    except OSError as e:
        raise IOError(f"Failed to create output directories: {e}")

    # Find all Arrow files
    arrow_files = glob.glob(os.path.join(input_dir, "**", "*.arrow"), recursive=True)

    if not arrow_files:
        raise ValueError(f"No Arrow files found in {input_dir}")

    print(f"Found {len(arrow_files)} Arrow files")
    print(f"First 5 files: {arrow_files[:5]}")

    # Process each Arrow file
    for i, arrow_file in enumerate(arrow_files):
        print(f"\nProcessing file {i + 1}/{len(arrow_files)}: {arrow_file}")
        process_arrow_file(arrow_file, audio_output_dir, text_output_dir, i)

    print("\nCompleted resegmenting all data")


if __name__ == "__main__":
    Fire(main)
