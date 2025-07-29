"""
Audio language assignment for JSONL files.

This module processes JSONL files in a directory and assigns audio language tags based on
language identification mappings. It combines all available mapping files and
applies the language assignments in parallel to all JSONL files found.

Features:
- Parallel processing of all JSONL files in input directory
- Automatic discovery and combination of all mapping files
- Missing language tracking and logging
- Progress tracking for large datasets
- Fallback to English for missing mappings
- Generic file processing without shard-specific constraints

The script processes all JSONL.gz files found in the input directory and combines
all mapping files found in the mapping directory for comprehensive coverage.
"""

import glob
import gzip
import json
import multiprocessing
import os
from collections import OrderedDict
from functools import partial
from itertools import repeat
from typing import Dict, List, Tuple, Optional, Any, Union

from fire import Fire
from tqdm import tqdm

# Type aliases for better readability
FilePath = str
VideoID = str
LanguageCode = str
LanguageMapping = Dict[VideoID, LanguageCode]


def load_language_mapping(mapping_file_path: FilePath) -> LanguageMapping:
    """
    Load language mapping from a compressed JSON file.

    Args:
        mapping_file_path: Path to the compressed JSON file containing language mappings

    Returns:
        Dictionary mapping video IDs to language codes

    Raises:
        FileNotFoundError: If the mapping file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        IOError: If file cannot be read
    """
    try:
        with gzip.open(mapping_file_path, "rt", encoding="utf-8") as f:
            language_mapping = json.load(f)
        return language_mapping
    except FileNotFoundError:
        raise FileNotFoundError(f"Language mapping file not found: {mapping_file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in mapping file {mapping_file_path}: {str(e)}", e.doc, e.pos
        )
    except Exception as e:
        raise IOError(f"Failed to load language mapping from {mapping_file_path}: {e}")


def combine_language_mappings(mapping_dir: FilePath) -> LanguageMapping:
    """
    Discover and combine all language mapping files in a directory.

    Args:
        mapping_dir: Directory containing language mapping files

    Returns:
        Combined dictionary mapping video IDs to language codes

    Raises:
        FileNotFoundError: If mapping directory doesn't exist or no files found
        IOError: If mapping files cannot be read
    """
    if not os.path.exists(mapping_dir):
        raise FileNotFoundError(f"Mapping directory not found: {mapping_dir}")

    # Find all JSON.gz files in the mapping directory
    mapping_pattern = os.path.join(mapping_dir, "*.json.gz")
    mapping_files = glob.glob(mapping_pattern)

    if not mapping_files:
        raise FileNotFoundError(f"No JSON.gz mapping files found in {mapping_dir}")

    print(f"Found {len(mapping_files)} mapping files to combine")

    # Combine all mappings
    combined_mapping = {}
    for mapping_file in tqdm(mapping_files, desc="Loading mapping files"):
        try:
            mapping = load_language_mapping(mapping_file)
            combined_mapping.update(mapping)
        except Exception as e:
            print(f"Warning: Failed to load mapping file {mapping_file}: {e}")

    print(f"Combined mapping contains {len(combined_mapping)} video IDs")
    return combined_mapping


def load_jsonl_data(jsonl_file_path: FilePath) -> List[Dict[str, Any]]:
    """
    Load data from a compressed JSONL file.

    Args:
        jsonl_file_path: Path to the compressed JSONL file

    Returns:
        List of JSON objects from the file

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        IOError: If file cannot be read
    """
    try:
        with gzip.open(jsonl_file_path, "rt", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"JSONL file not found: {jsonl_file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in file {jsonl_file_path}: {str(e)}", e.doc, e.pos
        )
    except Exception as e:
        raise IOError(f"Failed to load JSONL data from {jsonl_file_path}: {e}")


def save_processed_jsonl(data: List[Dict[str, Any]], output_path: FilePath) -> None:
    """
    Save processed JSONL data to a compressed file.

    Args:
        data: List of JSON objects to save
        output_path: Path where to save the processed data

    Raises:
        IOError: If file cannot be written
        json.JSONEncodeError: If data cannot be serialized to JSON
    """
    try:
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    except Exception as e:
        raise IOError(f"Failed to save processed JSONL to {output_path}: {e}")


def log_missing_language(
    video_id: VideoID,
    jsonl_file_path: FilePath,
    missing_log_path: FilePath,
) -> None:
    """
    Log information about a video with missing language mapping.

    Args:
        video_id: ID of the video missing language mapping
        jsonl_file_path: Path to the JSONL file containing the video
        missing_log_path: Path to the log file for missing mappings

    Raises:
        IOError: If log file cannot be written
    """
    try:
        with open(missing_log_path, "a", encoding="utf-8") as f:
            f.write(f"{video_id}, {jsonl_file_path}\n")
    except Exception as e:
        raise IOError(f"Failed to log missing language mapping: {e}")


def assign_audio_language_to_file(
    output_dir: FilePath,
    jsonl_file_path: FilePath,
    language_mapping: LanguageMapping,
    default_language: LanguageCode = "en",
) -> None:
    """
    Assign audio language tags to all entries in a single JSONL file.

    This function loads a JSONL file and assigns audio language tags to each video
    entry based on the provided language mapping. Videos without language mappings
    are assigned the default language and logged for tracking.

    Args:
        output_dir: Directory to save the processed file
        jsonl_file_path: Path to the input JSONL file
        language_mapping: Dictionary mapping video IDs to language codes
        default_language: Language code to use when no mapping is found (default: "en")

    Raises:
        FileNotFoundError: If JSONL file doesn't exist
        IOError: If file operations fail
    """
    missing_log_path = os.path.join(output_dir, "missing_audio_lang.txt")

    try:
        # Load JSONL data
        jsonl_data = load_jsonl_data(jsonl_file_path)

        # Process each entry in the file
        for entry in jsonl_data:
            video_id = entry.get("id")

            if not video_id:
                continue  # Skip entries without ID

            if video_id in language_mapping:
                entry["audio_lang"] = language_mapping[video_id]
            else:
                entry["audio_lang"] = default_language
                log_missing_language(video_id, jsonl_file_path, missing_log_path)

        # Save processed file
        output_filename = os.path.basename(jsonl_file_path)
        output_path = os.path.join(output_dir, output_filename)
        save_processed_jsonl(jsonl_data, output_path)

    except Exception as e:
        raise IOError(f"Failed to process JSONL file {jsonl_file_path}: {e}")


def process_file_wrapper(
    args: Tuple[FilePath, FilePath], language_mapping: Union[LanguageMapping, Any]
) -> None:
    """
    Wrapper function for parallel processing of JSONL files.

    This function unpacks arguments and calls the main file processing function.
    It's designed to work with multiprocessing.Pool.imap_unordered.

    Args:
        args: Tuple of (output_dir, jsonl_file_path)
        language_mapping: Dictionary mapping video IDs to language codes

    Raises:
        Exception: Any exception from the underlying file processing
    """
    output_dir, jsonl_file_path = args
    assign_audio_language_to_file(output_dir, jsonl_file_path, language_mapping)


def discover_jsonl_files(input_dir: FilePath) -> List[FilePath]:
    """
    Discover all JSONL.gz files in the input directory.

    Args:
        input_dir: Directory to search for JSONL files

    Returns:
        List of paths to discovered JSONL files

    Raises:
        FileNotFoundError: If input directory doesn't exist or no files found
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    jsonl_pattern = os.path.join(input_dir, "*.jsonl.gz")
    jsonl_files = glob.glob(jsonl_pattern)

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL.gz files found in {input_dir}")

    return sorted(jsonl_files)


def main(
    input_dir: FilePath,
    output_dir: FilePath,
    audio_lang_mapping_dir: FilePath,
    num_processes: Optional[int] = None,
) -> None:
    """
    Main entry point for audio language assignment pipeline.

    This function orchestrates the entire process of assigning audio language tags
    to JSONL files based on language mappings. It discovers all JSONL files in the
    input directory, combines all mapping files, and processes files in parallel.

    Args:
        input_dir: Directory containing input JSONL.gz files
        output_dir: Directory to save processed files with language tags
        audio_lang_mapping_dir: Directory containing language mapping files
        num_processes: Number of processes for parallel processing (default: CPU count)

    Raises:
        FileNotFoundError: If directories or files don't exist
        ValueError: If invalid parameters
        RuntimeError: If processing fails
    """
    # Validate inputs
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not os.path.exists(audio_lang_mapping_dir):
        raise FileNotFoundError(
            f"Audio language mapping directory not found: {audio_lang_mapping_dir}"
        )

    # Set default number of processes
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Audio Language Assignment Pipeline")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Mapping directory: {audio_lang_mapping_dir}")
    print(f"Processes: {num_processes}")

    try:
        # Combine all language mappings
        print("Combining language mappings...")
        language_mapping = combine_language_mappings(audio_lang_mapping_dir)

        # Discover JSONL files
        print("Discovering JSONL files...")
        jsonl_files = discover_jsonl_files(input_dir)
        print(f"Found {len(jsonl_files)} JSONL files")

        # Process files in parallel
        print(f"Processing {len(jsonl_files)} JSONL files...")

        # Create shared dictionary for multiprocessing
        manager = multiprocessing.Manager()
        shared_mapping = manager.dict(language_mapping)

        # Create partial function with shared mapping
        process_function = partial(
            process_file_wrapper, language_mapping=shared_mapping
        )

        # Prepare arguments for parallel processing
        process_args = [(output_dir, jsonl_file) for jsonl_file in jsonl_files]

        # Execute parallel processing
        with multiprocessing.Pool(processes=num_processes) as pool:
            list(
                tqdm(
                    pool.imap_unordered(process_function, process_args),
                    total=len(process_args),
                    desc="Processing files",
                )
            )

        print(f"Successfully processed {len(jsonl_files)} JSONL files")
        print(f"Results saved to: {output_dir}")

        # Check for missing language log
        missing_log_path = os.path.join(output_dir, "missing_audio_lang.txt")
        if os.path.exists(missing_log_path):
            with open(missing_log_path, "r", encoding="utf-8") as f:
                missing_count = len(f.readlines())
            print(
                f"Warning: {missing_count} videos had missing language mappings (logged to missing_audio_lang.txt)"
            )

    except Exception as e:
        raise RuntimeError(f"Audio language assignment failed: {e}")


if __name__ == "__main__":
    Fire(main)
