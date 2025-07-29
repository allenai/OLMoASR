"""
Text to JSONL Conversion Script

This script converts transcript files (various formats) to compressed JSONL format.
It processes transcript files and extracts metadata including content, length, and 
associated audio file paths.

The script supports:
- Single file processing
- Directory processing (all transcript files)
- List of files processing
- Parallel processing for performance
- Compressed JSONL output

Supported transcript formats:
- .srt (SubRip Subtitle)
- .vtt (WebVTT)
- .txt (Plain text)
- Other formats supported by TranscriptReader

Usage:
    python text_to_jsonl.py --input_path="/path/to/transcripts" --output_path="/path/to/output.jsonl.gz"
    python text_to_jsonl.py --input_path="/path/to/file.srt" --output_path="/path/to/output.jsonl.gz"
    python text_to_jsonl.py --input_path='["/file1.srt", "/file2.vtt"]' --output_path="/path/to/output.jsonl.gz"

Dependencies:
    - olmoasr: For transcript reading and processing utilities
    - fire: Command-line interface
    - tqdm: Progress tracking

Security Note:
    This script contains hardcoded AWS credentials which should be removed
    or moved to environment variables/configuration files in production.
"""

import multiprocessing
import os
import json
import gzip
import glob
from typing import List, Union, Dict, Any, Optional

from tqdm import tqdm
from fire import Fire

from olmoasr.utils import calculate_difference, TranscriptReader

# Type aliases for better readability
FilePath = str
TranscriptDict = Dict[str, Any]
FileList = List[FilePath]


def extract_file_info(file_path: FilePath) -> Dict[str, str]:
    """
    Extract file information from file path.

    Args:
        file_path: Path to the transcript file

    Returns:
        Dictionary containing file extension, base name, and directory
    """
    ext = file_path.split(".")[-1]
    base_name = file_path.split("/")[-1].split(".")[0]
    base_path = file_path.split(".")[0]

    return {"extension": ext, "base_name": base_name, "base_path": base_path}


def calculate_transcript_length(transcript_start: Any, transcript_end: Any) -> float:
    """
    Calculate the total length of a transcript in seconds.

    Args:
        transcript_start: Start times for transcript segments (format depends on TranscriptReader)
        transcript_end: End times for transcript segments (format depends on TranscriptReader)

    Returns:
        Total length in seconds, or 0 if calculation fails
    """
    if not transcript_start or not transcript_end:
        return 0.0

    try:
        # Calculate difference returns time in milliseconds, convert to seconds
        return calculate_difference(transcript_start, transcript_end) / 1000.0
    except (ValueError, IndexError, TypeError) as e:
        print(f"Warning: Failed to calculate transcript length: {e}")
        return 0.0


def process_transcript_file(
    file_path: FilePath, audio_extension: str = ".m4a"
) -> TranscriptDict:
    """
    Process a single transcript file and extract metadata.

    Args:
        file_path: Path to the transcript file
        audio_extension: Extension for associated audio file (default: .m4a)

    Returns:
        Dictionary containing transcript metadata

    Raises:
        FileNotFoundError: If the transcript file doesn't exist
        ValueError: If the transcript format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcript file not found: {file_path}")

    try:
        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract file information
        file_info = extract_file_info(file_path)

        # Parse transcript using TranscriptReader
        reader = TranscriptReader(
            file_path=None, transcript_string=content, ext=file_info["extension"]
        )
        transcript, transcript_start, transcript_end = reader.read()

        # Calculate transcript length
        length = calculate_transcript_length(transcript_start, transcript_end)

        # Construct audio file path
        audio_file = file_info["base_path"] + audio_extension

        return {
            "subtitle_file": file_path,
            "content": content,
            "length": length,
            "audio_file": audio_file,
            "id": file_info["base_name"],
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return a minimal dictionary for failed files
        file_info = extract_file_info(file_path)
        return {
            "subtitle_file": file_path,
            "content": "",
            "length": 0.0,
            "audio_file": file_info["base_path"] + audio_extension,
            "id": file_info["base_name"],
        }


def parallel_process_file(file_path: FilePath) -> TranscriptDict:
    """
    Wrapper function for parallel processing of transcript files.

    Args:
        file_path: Path to the transcript file

    Returns:
        Dictionary containing transcript metadata
    """
    return process_transcript_file(file_path)


def discover_transcript_files(input_path: FilePath, pattern: str = "*.*t") -> FileList:
    """
    Discover transcript files in a directory.

    Args:
        input_path: Directory path to search
        pattern: File pattern to match (default: "*.*t" for .srt, .vtt, .txt, etc.)

    Returns:
        List of transcript file paths

    Raises:
        ValueError: If no transcript files are found
    """
    if not os.path.isdir(input_path):
        raise ValueError(f"Input path is not a directory: {input_path}")

    files = glob.glob(os.path.join(input_path, pattern))

    if not files:
        raise ValueError(
            f"No transcript files found in {input_path} with pattern {pattern}"
        )

    return sorted(files)


def prepare_input_files(input_path: Union[str, List[str]]) -> FileList:
    """
    Prepare list of input files from various input types.

    Args:
        input_path: Single file path, directory path, or list of file paths

    Returns:
        List of transcript file paths

    Raises:
        ValueError: If input is invalid or no files are found
    """
    if isinstance(input_path, str):
        if os.path.isdir(input_path):
            # Directory: discover transcript files
            return discover_transcript_files(input_path)
        elif os.path.isfile(input_path):
            # Single file
            return [input_path]
        else:
            raise ValueError(f"Input path does not exist: {input_path}")
    elif isinstance(input_path, list):
        # List of files: validate each exists
        for file_path in input_path:
            if not os.path.isfile(file_path):
                raise ValueError(f"File does not exist: {file_path}")
        return input_path
    else:
        raise ValueError(f"Invalid input_path type: {type(input_path)}")


def write_jsonl_output(
    transcript_dicts: List[TranscriptDict], output_path: FilePath
) -> None:
    """
    Write transcript dictionaries to compressed JSONL file.

    Args:
        transcript_dicts: List of transcript metadata dictionaries
        output_path: Path for output JSONL file

    Raises:
        IOError: If writing to output file fails
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        print(f"Writing {len(transcript_dicts)} entries to {output_path}")

        # Write to compressed JSONL file
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            for transcript_dict in transcript_dicts:
                json_line = json.dumps(transcript_dict, ensure_ascii=False)
                f.write(json_line + "\n")

        print(f"Successfully wrote {len(transcript_dicts)} entries")

    except Exception as e:
        raise IOError(f"Failed to write output file {output_path}: {e}")


def process_transcripts_parallel(
    input_files: FileList, num_processes: Optional[int] = None
) -> List[TranscriptDict]:
    """
    Process transcript files in parallel.

    Args:
        input_files: List of transcript file paths
        num_processes: Number of processes to use (default: system CPU count)

    Returns:
        List of transcript metadata dictionaries
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    print(f"Processing {len(input_files)} files using {num_processes} processes")

    with multiprocessing.Pool(processes=num_processes) as pool:
        transcript_dicts = list(
            tqdm(
                pool.imap_unordered(parallel_process_file, input_files),
                total=len(input_files),
                desc="Processing transcripts",
            )
        )

    return transcript_dicts


def text_to_jsonl(
    input_path: Union[str, List[str]],
    output_path: str,
    num_processes: Optional[int] = None,
) -> None:
    """
    Convert transcript files to compressed JSONL format.

    This function processes transcript files (single file, directory, or list of files)
    and converts them to a compressed JSONL format with extracted metadata.

    Args:
        input_path: Path to transcript file, directory containing transcripts,
                   or list of transcript file paths
        output_path: Path for output compressed JSONL file
        num_processes: Number of processes for parallel processing (default: system CPU count)

    Raises:
        ValueError: If input_path is invalid or no files are found
        IOError: If output file cannot be written
        FileNotFoundError: If input files don't exist

    Example:
        >>> # Process single file
        >>> text_to_jsonl("transcript.srt", "output.jsonl.gz")

        >>> # Process directory
        >>> text_to_jsonl("/path/to/transcripts/", "output.jsonl.gz")

        >>> # Process list of files
        >>> text_to_jsonl(["file1.srt", "file2.vtt"], "output.jsonl.gz")
    """
    # Prepare input files
    input_files = prepare_input_files(input_path)
    print(f"Found {len(input_files)} transcript files to process")

    # Process transcripts in parallel
    transcript_dicts = process_transcripts_parallel(input_files, num_processes)

    # Display processing statistics
    valid_transcripts = [d for d in transcript_dicts if d["length"] > 0]
    print(f"Processing complete:")
    print(f"  Total files: {len(transcript_dicts)}")
    print(f"  Valid transcripts: {len(valid_transcripts)}")
    print(f"  Failed/empty: {len(transcript_dicts) - len(valid_transcripts)}")

    if transcript_dicts:
        print(f"Sample entry: {transcript_dicts[0]}")

    # Write output
    write_jsonl_output(transcript_dicts, output_path)


if __name__ == "__main__":
    Fire(text_to_jsonl)
