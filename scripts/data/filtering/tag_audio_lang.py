"""
Audio language identification for transcript segments using SpeechBrain models.

This script processes audio segments from transcript data and identifies the language
of each segment using pre-trained language identification models. It assigns language
tags to segments based on audio content analysis, processing all files in a directory
as a unified dataset.
"""

import os
import glob
import json
import gzip
import multiprocessing
from collections import defaultdict
from itertools import chain
from typing import List, Dict, Tuple, Any, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from fire import Fire

# Optional imports with fallback handling
try:
    from speechbrain.inference.classifiers import EncoderClassifier  # type: ignore

    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    EncoderClassifier = None

try:
    from whisper import audio  # type: ignore

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    audio = None


class AudioLanguageDataset(Dataset):
    """
    Dataset for loading audio segments and associated metadata for language identification.

    This dataset handles loading audio files, preprocessing them for the language
    identification model, and extracting relevant metadata from transcript segments.
    """

    def __init__(self, data: List[Dict[str, Any]]):
        """
        Initialize the dataset.

        Args:
            data: List of transcript segment dictionaries containing audio file paths
                 and metadata

        Raises:
            ValueError: If data is empty or invalid
            ImportError: If required dependencies are not available
        """
        if not data:
            raise ValueError("Dataset cannot be empty")

        if not WHISPER_AVAILABLE or audio is None:
            raise ImportError("Whisper audio processing not available")

        self.data = data

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray, str]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (video_id, audio_array, segment_content)

        Raises:
            IndexError: If idx is out of range
            FileNotFoundError: If audio file doesn't exist
        """
        if idx >= len(self.data):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.data)}"
            )

        sample = self.data[idx]

        # Extract metadata
        video_id = sample.get("id", "")
        content = sample.get("seg_content", "")

        # Get audio file path and convert path format
        audio_file = sample.get("audio_file", "")
        audio_file = audio_file.replace("ow_full", "ow_seg")

        # Load and preprocess audio
        audio_arr = self._load_and_preprocess_audio(audio_file)

        return video_id, audio_arr, content

    def _load_and_preprocess_audio(self, audio_file: str) -> np.ndarray:
        """
        Load and preprocess audio file for language identification.

        Args:
            audio_file: Path to the audio file (.npy format)

        Returns:
            Preprocessed audio array

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file is invalid or corrupted
        """
        try:
            # Load audio array and normalize
            audio_arr = np.load(audio_file).astype(np.float32) / 32768.0

            # Pad or trim to standard length for model input
            if WHISPER_AVAILABLE and audio is not None:
                processed_audio = audio.pad_or_trim(audio_arr)
                # Ensure we return a numpy array
                if isinstance(processed_audio, np.ndarray):
                    audio_arr = processed_audio
                else:
                    # Convert tensor to numpy if needed
                    audio_arr = np.array(processed_audio)

            return audio_arr

        except FileNotFoundError:
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        except Exception as e:
            raise ValueError(f"Failed to load audio file {audio_file}: {e}")


def load_compressed_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a compressed JSONL file.

    Args:
        file_path: Path to the .jsonl.gz file

    Returns:
        List of JSON objects parsed from the file

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in file {file_path}: {str(e)}", e.doc, e.pos
        )


def assign_language_tag(
    sample_dict: Dict[str, Any], language_id: str
) -> Dict[str, Any]:
    """
    Add language tag to a sample dictionary.

    Args:
        sample_dict: Original sample dictionary
        language_id: Identified language code

    Returns:
        Updated sample dictionary with language tag
    """
    sample_dict["audio_lang"] = language_id
    return sample_dict


def identify_segment_languages(
    model: Any, dataloader: DataLoader, device: torch.device
) -> List[Tuple[str, str]]:
    """
    Identify languages for audio segments using the classification model.

    Args:
        model: Pre-trained SpeechBrain language identification model
        dataloader: DataLoader containing audio segments
        device: PyTorch device for computation

    Returns:
        List of (video_id, language_id) tuples for segments with content

    Raises:
        RuntimeError: If model inference fails
    """
    predicted_languages = []

    print(f"Processing {len(dataloader)} batches for language identification...")

    try:
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Identifying languages")
        ):
            video_ids, audio_arrays, contents = batch

            # Move audio to device
            audio_arrays = audio_arrays.to(device)

            # Run language identification
            results = model.classify_batch(audio_arrays)

            # Extract language predictions for segments with content
            for i, classification_result in enumerate(results[3]):
                if contents[i] != "":  # Only process segments with content
                    language_id = classification_result.split(": ")[0]
                    predicted_languages.append((video_ids[i], language_id))

        return predicted_languages

    except Exception as e:
        raise RuntimeError(f"Language identification failed: {e}")


def aggregate_video_languages(
    predicted_languages: List[Tuple[str, str]]
) -> Dict[str, str]:
    """
    Aggregate language predictions by video ID using majority voting.

    Args:
        predicted_languages: List of (video_id, language_id) predictions

    Returns:
        Dictionary mapping video_id to most frequent language
    """
    # Group predictions by video ID
    video_language_counts = defaultdict(list)
    for video_id, language_id in predicted_languages:
        video_language_counts[video_id].append(language_id)

    # Determine most frequent language for each video
    video_languages = {
        video_id: max(languages, key=languages.count)
        for video_id, languages in video_language_counts.items()
    }

    return video_languages


def process_data(
    input_dir: str,
    output_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> None:
    """
    Process audio language identification for transcript segments from a directory.

    This function loads transcript data from all .jsonl.gz files in the input directory,
    processes the associated audio segments through a language identification model,
    and saves the results as language mappings per video. All files in the directory
    are processed as a unified dataset.

    Args:
        input_dir: Directory containing input .jsonl.gz files
        output_dir: Directory to save language identification results
        batch_size: Batch size for model inference (default: 32)
        num_workers: Number of worker processes for data loading (default: 4)

    Raises:
        ImportError: If required dependencies are not available
        FileNotFoundError: If source directory or files don't exist
        ValueError: If invalid parameters are provided
        RuntimeError: If processing fails
    """
    # Validate dependencies
    if not SPEECHBRAIN_AVAILABLE or EncoderClassifier is None:
        raise ImportError(
            "SpeechBrain is required but not available. Please install SpeechBrain."
        )

    if not WHISPER_AVAILABLE or audio is None:
        raise ImportError(
            "Whisper is required but not available. Please install Whisper."
        )

    # Validate parameters
    if batch_size <= 0 or num_workers < 0:
        raise ValueError(
            "batch_size must be positive and num_workers must be non-negative"
        )

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Source directory not found: {input_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Directory processing configuration:")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers}")

    # Find all .jsonl.gz files in the directory
    input_paths = sorted(glob.glob(os.path.join(input_dir, "*.jsonl.gz")))

    if not input_paths:
        raise FileNotFoundError(f"No .jsonl.gz files found in {input_dir}")

    print(f"Found {len(input_paths)} files to process")

    # Load data from all files in parallel
    print("Loading data from directory...")
    with multiprocessing.Pool() as pool:
        data_list = list(
            tqdm(
                pool.imap_unordered(load_compressed_jsonl, input_paths),
                total=len(input_paths),
                desc="Loading files",
            )
        )

    # Combine all data from the directory into a single dataset
    all_data = list(chain(*data_list))
    print(f"Loaded {len(all_data)} total samples from directory")

    # Initialize language identification model
    print("Initializing language identification model...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="tmp",
            run_opts={"device": str(device)},
        )
        print("Model initialized successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize language identification model: {e}")

    # Create dataset and dataloader for unified processing
    print("Preparing unified dataset for processing...")
    try:
        dataset = AudioLanguageDataset(all_data)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
        print(f"Created unified dataset with {len(dataset)} samples")
    except Exception as e:
        raise RuntimeError(f"Failed to create dataset: {e}")

    # Perform language identification on the unified dataset
    print("Running language identification on unified dataset...")
    predicted_languages = identify_segment_languages(model, dataloader, device)

    # Aggregate results by video ID
    print("Aggregating results by video...")
    video_languages = aggregate_video_languages(predicted_languages)

    # Print comprehensive statistics
    print(f"Directory processing results:")
    print(f"  Files processed: {len(input_paths)}")
    print(f"  Segments processed: {len(predicted_languages)}")
    print(f"  Unique videos identified: {len(video_languages)}")

    # Language distribution statistics
    language_counts = defaultdict(int)
    for lang in video_languages.values():
        language_counts[lang] += 1

    print(f"  Language distribution:")
    for lang, count in sorted(
        language_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"    {lang}: {count} videos")

    # Save results
    output_path = os.path.join(output_dir, "ids_to_lang.json.gz")

    print(f"Saving results to {output_path}...")
    try:
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            json.dump(video_languages, f, indent=2)
        print(f"Successfully saved language mappings to {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save results: {e}")


if __name__ == "__main__":
    Fire(
        {
            "assign_language_tag": assign_language_tag,
            "identify_segment_languages": identify_segment_languages,
            "aggregate_video_languages": aggregate_video_languages,
            "process_data": process_data,
        }
    )