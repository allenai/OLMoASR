"""
Parallel reservoir sampling for JSONL data files.

This module implements a distributed reservoir sampling algorithm for efficiently
sampling from large collections of JSONL files. It uses multiprocessing to
parallelize the sampling process across multiple worker processes.

The implementation uses an approximate reservoir sampling approach:
- Files are partitioned into chunks for parallel processing
- Each worker process performs reservoir sampling on its chunk
- Results are merged to produce the final sample

Note: This is not perfect reservoir sampling due to the parallel nature,
but provides good approximation with significant performance benefits.
For perfect reservoir sampling, use num_cpus=1.

Features:
- Parallel processing for large datasets
- Memory-efficient streaming processing
- Configurable reservoir size
- Percentile analysis of sampled values
- Progress tracking with tqdm
"""

import argparse
import glob
import json
import os
import random
import tempfile
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Lock, Manager, Pool, Process, Queue
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from tqdm import tqdm

# Optional imports with fallback handling
try:
    from smart_open import open as smart_open  # type: ignore
    SMART_OPEN_AVAILABLE = True
except ImportError:
    SMART_OPEN_AVAILABLE = False
    smart_open = open  # Fallback to built-in open

try:
    import tabulate  # type: ignore
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    tabulate = None


# Type aliases for better readability
FilePath = str
JSONValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]
PercentileList = List[JSONValue]


class SharedCounter:
    """
    Thread-safe counter for multiprocessing environments.

    Uses a Manager to create a shared value that can be safely
    incremented across multiple processes with proper locking.
    """

    def __init__(self, initial_value: int = 0):
        """
        Initialize the shared counter.

        Args:
            initial_value: Starting value for the counter (default: 0)
        """
        self.val = Manager().Value("i", initial_value)
        self.lock = Lock()

    def increment(self) -> int:
        """
        Atomically increment the counter by 1.

        Returns:
            The new value after incrementing

        Note:
            This operation is thread-safe across multiple processes
        """
        with self.lock:
            self.val.value += 1
        return self.val.value

    @property
    def value(self) -> int:
        """Get the current counter value."""
        return self.val.value


def load_jsonl_file(file_path: FilePath) -> List[Dict[str, Any]]:
    """
    Load and parse a JSONL file.

    Args:
        file_path: Path to the JSONL file to load

    Returns:
        List of parsed JSON objects

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        IOError: If file cannot be read
    """
    try:
        if SMART_OPEN_AVAILABLE:
            with smart_open(file_path, "rb") as f:
                content = f.read()
        else:
            with open(file_path, "rb") as f:
                content = f.read()

        lines = content.decode("utf-8").splitlines()
        return [json.loads(line.strip()) for line in lines if line.strip()]

    except FileNotFoundError:
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e}")
    except Exception as e:
        raise IOError(f"Failed to read file {file_path}: {e}")


def reservoir_sample_chunk(
    file_chunk: List[FilePath],
    key: str,
    chunk_reservoir_size: int,
    counter: SharedCounter,
    result_queue: Queue,
) -> None:
    """
    Perform reservoir sampling on a chunk of files.

    This function processes a subset of files and maintains a reservoir
    sample of the specified size. It extracts values using the given key
    and applies the reservoir sampling algorithm.

    Args:
        file_chunk: List of file paths to process
        key: JSON key to extract values from
        chunk_reservoir_size: Maximum size of the reservoir for this chunk
        counter: Shared counter for tracking progress across processes
        result_queue: Queue to store the temporary file with results

    Raises:
        KeyError: If the specified key is not found in JSON objects
        ValueError: If chunk_reservoir_size is not positive
        IOError: If file operations fail
    """
    if chunk_reservoir_size <= 0:
        raise ValueError("chunk_reservoir_size must be positive")

    # Create temporary file to store results
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    documents_processed = 0
    reservoir = []

    try:
        for file_path in file_chunk:
            try:
                lines = load_jsonl_file(file_path)

                for line in lines:
                    if key not in line:
                        continue  # Skip entries without the required key

                    value = line[key]

                    # Apply reservoir sampling algorithm
                    if len(reservoir) < chunk_reservoir_size:
                        reservoir.append(value)
                    else:
                        # Replace random element with probability k/n
                        random_index = random.randint(0, documents_processed)
                        if random_index < chunk_reservoir_size:
                            reservoir[random_index] = value

                    documents_processed += 1

                # Update progress counter
                counter.increment()

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

        # Save reservoir to temporary file
        temp_file.write(json.dumps(reservoir).encode("utf-8"))
        temp_file.flush()
        temp_file.close()

        # Add temp file path to result queue
        result_queue.put(temp_file.name)

    except Exception as e:
        # Clean up temp file if error occurs
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise IOError(f"Failed to process chunk: {e}")


def collect_and_merge_results(result_queue: Queue) -> List[JSONValue]:
    """
    Collect and merge reservoir sampling results from worker processes.

    Args:
        result_queue: Queue containing paths to temporary result files

    Returns:
        Merged list of sampled values from all worker processes

    Raises:
        IOError: If temporary files cannot be read or deleted
        json.JSONDecodeError: If temporary files contain invalid JSON
    """
    merged_reservoir = []

    while not result_queue.empty():
        temp_file_path = result_queue.get()

        try:
            with open(temp_file_path, "rb") as f:
                chunk_reservoir = json.loads(f.read().decode("utf-8"))
                merged_reservoir.extend(chunk_reservoir)
        except Exception as e:
            print(f"Error reading temporary file {temp_file_path}: {e}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError as e:
                print(f"Warning: Failed to delete temporary file {temp_file_path}: {e}")

    return merged_reservoir


def calculate_percentiles(values: List[JSONValue]) -> PercentileList:
    """
    Calculate percentiles from a list of values.

    Args:
        values: List of values to calculate percentiles for

    Returns:
        List of percentile values (0th through 99th percentile, plus maximum)

    Raises:
        ValueError: If values list is empty
    """
    if not values:
        raise ValueError("Cannot calculate percentiles from empty list")

    # Sort values for percentile calculation
    sorted_values = sorted(values)
    n = len(sorted_values)

    # Calculate percentiles (0-99th percentile + maximum)
    percentiles = []
    for i in range(100):
        index = min(round(i * n / 100), n - 1)
        percentiles.append(sorted_values[index])

    # Add maximum value as 100th percentile
    percentiles.append(sorted_values[-1])

    return percentiles


def partition_files_into_chunks(
    files: List[FilePath], num_chunks: int
) -> List[List[FilePath]]:
    """
    Partition files into approximately equal chunks for parallel processing.

    Args:
        files: List of file paths to partition
        num_chunks: Number of chunks to create

    Returns:
        List of file chunks, each containing a subset of the input files

    Raises:
        ValueError: If num_chunks is not positive
    """
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")

    chunks = [[] for _ in range(num_chunks)]

    for i, file_path in enumerate(files):
        chunk_index = i % num_chunks
        chunks[chunk_index].append(file_path)

    return chunks


def display_percentile_table(percentiles: PercentileList) -> None:
    """
    Display percentiles in a formatted table.

    Args:
        percentiles: List of percentile values to display
    """
    table_data = []
    for i, value in enumerate(percentiles[:-1]):  # Exclude the last (max) value
        table_data.append([f"{i}th", value])

    # Add maximum value
    table_data.append(["max", percentiles[-1]])

    if TABULATE_AVAILABLE and tabulate is not None:
        print(tabulate.tabulate(table_data, headers=["Percentile", "Value"]))
    else:
        # Fallback to simple formatting when tabulate is not available
        print(f"{'Percentile':<12} {'Value'}")
        print("-" * 30)
        for row in table_data:
            print(f"{row[0]:<12} {row[1]}")


def save_percentiles(percentiles: PercentileList, output_path: FilePath) -> None:
    """
    Save percentiles to a JSON file.

    Args:
        percentiles: List of percentile values to save
        output_path: Path where to save the percentiles file

    Raises:
        IOError: If file cannot be written
        json.JSONEncodeError: If percentiles cannot be serialized to JSON
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save percentiles as JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(percentiles, f, indent=2)

    except Exception as e:
        raise IOError(f"Failed to save percentiles to {output_path}: {e}")


def main(
    input_dir: FilePath,
    key: str,
    output_location: Optional[FilePath] = None,
    reservoir_size: int = 1_000_000,
    num_cpus: Optional[int] = None,
) -> None:
    """
    Perform parallel reservoir sampling on JSONL files in a directory.

    This function implements a distributed reservoir sampling algorithm that:
    1. Discovers all JSONL files in the input directory
    2. Partitions files across multiple worker processes
    3. Performs reservoir sampling on each partition
    4. Merges results and calculates percentiles
    5. Displays and optionally saves the results

    Args:
        input_dir: Directory containing JSONL files to sample from
        key: JSON key to extract values for sampling
        output_location: Optional path to save percentile results
        reservoir_size: Total number of samples to collect (default: 1,000,000)
        num_cpus: Number of CPU cores to use (default: all available)

    Raises:
        FileNotFoundError: If input directory doesn't exist
        ValueError: If invalid parameters are provided
        RuntimeError: If sampling process fails
    """
    start_time = time.time()

    # Validate inputs
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if reservoir_size <= 0:
        raise ValueError("reservoir_size must be positive")

    # Set default number of CPUs
    if num_cpus is None:
        num_cpus = os.cpu_count() or 1

    if num_cpus <= 0:
        raise ValueError("num_cpus must be positive")

    # Discover JSONL files
    file_pattern = os.path.join(input_dir, "**/*.jsonl*")
    files = glob.glob(file_pattern, recursive=True)

    if not files:
        raise FileNotFoundError(f"No JSONL files found in {input_dir}")

    print(f"Starting reservoir sampling:")
    print(f"  Key: {key}")
    print(f"  Files: {len(files)}")
    print(f"  Reservoir size: {reservoir_size:,}")
    print(f"  CPU cores: {num_cpus}")

    # Partition files into chunks for parallel processing
    file_chunks = partition_files_into_chunks(files, num_cpus)
    chunk_reservoir_size = max(1, round(reservoir_size / num_cpus))

    # Initialize shared resources
    counter = SharedCounter()
    result_queue = Queue()

    # Start worker processes
    processes = []
    for i, chunk in enumerate(file_chunks):
        if not chunk:  # Skip empty chunks
            continue

        process = Process(
            target=reservoir_sample_chunk,
            args=(chunk, key, chunk_reservoir_size, counter, result_queue),
        )
        processes.append(process)
        process.start()

    # Monitor progress
    with tqdm(total=len(files), desc="Processing files") as pbar:
        last_count = 0

        while any(p.is_alive() for p in processes):
            current_count = counter.value
            if current_count > last_count:
                pbar.update(current_count - last_count)
                last_count = current_count
            time.sleep(0.1)

        # Final update for any remaining progress
        final_count = counter.value
        if final_count > last_count:
            pbar.update(final_count - last_count)

    # Wait for all processes to complete
    for process in processes:
        process.join()

    # Collect and merge results
    print("Collecting and merging results...")
    try:
        merged_reservoir = collect_and_merge_results(result_queue)

        if not merged_reservoir:
            raise RuntimeError(f"No values found for key '{key}' in any files")

        # Calculate percentiles
        percentiles = calculate_percentiles(merged_reservoir)

        # Display results
        print(f"\nSampling completed in {time.time() - start_time:.2f} seconds")
        print(f"Collected {len(merged_reservoir):,} samples")
        print("\nPercentile Analysis:")
        display_percentile_table(percentiles)

        # Save results if output location is specified
        if output_location:
            save_percentiles(percentiles, output_location)
            print(f"\nResults saved to: {output_location}")

    except Exception as e:
        raise RuntimeError(f"Failed to process sampling results: {e}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace

    Raises:
        SystemExit: If argument parsing fails
    """
    parser = argparse.ArgumentParser(
        description="Perform parallel reservoir sampling on JSONL files",
        epilog="Example: python reservoir_sample.py --input-dir /data --key word_count --reservoir-size 100000",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing JSONL files to sample from",
    )

    parser.add_argument(
        "--key", type=str, required=True, help="JSON key to extract values for sampling"
    )

    parser.add_argument(
        "--output-loc",
        type=str,
        default=None,
        help="Optional path to save percentile results as JSON",
    )

    parser.add_argument(
        "--reservoir-size",
        type=int,
        default=1_000_000,
        help="Total number of samples to collect (default: 1,000,000)",
    )

    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPU cores to use (default: all available)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_arguments()

        main(
            input_dir=args.input_dir,
            key=args.key,
            output_location=args.output_loc,
            reservoir_size=args.reservoir_size,
            num_cpus=args.num_cpus,
        )

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
