"""
OLMoASR Evaluation Module

This module provides comprehensive evaluation functionality for automatic speech recognition (ASR) models,
supporting both short-form and long-form transcription tasks across multiple datasets.

Key Features:
    - Support for 20+ evaluation datasets (LibriSpeech, TEDLIUM, CORAAL, etc.)
    - Short-form and long-form transcription evaluation
    - Weights & Biases integration for experiment tracking
    - Bootstrap sampling for confidence intervals
    - Multi-modal audio processing pipeline
    - Extensible dataset loader architecture

Usage:
    # Short-form evaluation
    python eval.py short_form_eval --batch_size 32 --ckpt model.pt --eval_set librispeech_clean --log_dir logs/

    # Long-form evaluation  
    python eval.py long_form_eval --batch_size 8 --ckpt model.pt --eval_set tedlium --log_dir logs/

Classes:
    AudioSegment: Data class representing audio segments with timing
    AudioProcessor: Utility class for audio preprocessing operations
    TextCleaner: Text normalization and cleaning utilities
    BaseDatasetLoader: Abstract base class for dataset loaders
    DatasetFactory: Factory pattern for creating dataset loaders
    EvalDataset: PyTorch Dataset for evaluation data
    WandBLogger: Weights & Biases logging utilities

Functions:
    short_form_eval: Main evaluation function for short-form transcription
    long_form_eval: Main evaluation function for long-form transcription
"""

from typing import Literal, Optional, Tuple, Dict, Union
import os
import glob
import json
import re
import io
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import jiwer
from whisper import audio, DecodingOptions
from whisper.normalizers import EnglishTextNormalizer
from olmoasr import load_model
from fire import Fire
from tqdm import tqdm
from scripts.eval.get_eval_set import get_eval_set
from scripts.eval.gen_inf_ckpt import gen_inf_ckpt
import wandb
from torchaudio.datasets import TEDLIUM
import torchaudio
import csv
import subprocess
from dataclasses import dataclass
from abc import ABC, abstractmethod


# Constants
SAMPLE_RATE = 16000
DEFAULT_N_MELS = 80
HF_DATASETS = {
    "tedlium",
    "common_voice",
    "fleurs",
    "voxpopuli",
    "meanwhile",
    "rev16",
    "earnings21",
    "earnings22",
}
LONG_FORM_DATASETS = {
    "meanwhile",
    "rev16",
    "earnings21",
    "earnings22",
    "coraal",
    "kincaid46",
    "tedlium",
}


@dataclass
class AudioSegment:
    """Represents an audio segment with optional timing information.

    This data class encapsulates audio file paths along with optional start and end times
    for segmented audio processing, commonly used in datasets like CallHome and SwitchBoard.

    Attributes:
        file_path (str): Path to the audio file
        start_time (Optional[float]): Start time in seconds, None for full audio
        end_time (Optional[float]): End time in seconds, None for full audio

    Example:
        >>> segment = AudioSegment("audio.wav", 10.5, 25.3)
        >>> print(f"Duration: {segment.end_time - segment.start_time}s")
        Duration: 14.8s
    """

    file_path: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class AudioProcessor:
    """Handles common audio processing operations for ASR evaluation.

    This utility class provides static methods for loading, preprocessing, and transforming
    audio data for both short-form and long-form transcription tasks. It standardizes
    audio processing across different dataset formats and sources.

    Methods:
        load_and_preprocess_audio: Load and preprocess audio files with optional segmentation
        preprocess_hf_audio: Preprocess audio arrays from HuggingFace datasets
    """

    @staticmethod
    def load_and_preprocess_audio(
        audio_file: str,
        sr: int = SAMPLE_RATE,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        n_mels: int = DEFAULT_N_MELS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess audio file with optional time-based segmentation.

        Loads an audio file, optionally extracts a time segment, pads/trims to standard
        length, and computes mel-spectrogram features for ASR model input.

        Args:
            audio_file (str): Path to the audio file to load
            sr (int, optional): Target sampling rate. Defaults to SAMPLE_RATE (16000).
            start_time (Optional[float], optional): Start time in seconds for segmentation.
                Defaults to None.
            end_time (Optional[float], optional): End time in seconds for segmentation.
                Defaults to None.
            n_mels (int, optional): Number of mel-spectrogram bins. Defaults to DEFAULT_N_MELS.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - audio_arr: Preprocessed audio waveform as float32 array
                - mel_spec: Log mel-spectrogram features for model input

        Example:
            >>> audio_arr, mel_spec = AudioProcessor.load_and_preprocess_audio(
            ...     "speech.wav", start_time=10.0, end_time=20.0
            ... )
            >>> print(f"Audio shape: {audio_arr.shape}, Mel shape: {mel_spec.shape}")
        """
        audio_arr = audio.load_audio(audio_file, sr=sr)
        if start_time is not None and end_time is not None:
            audio_arr = audio_arr[int(start_time * sr) : int(end_time * sr)]
        audio_arr = audio.pad_or_trim(audio_arr)
        audio_arr = audio_arr.astype(np.float32)
        mel_spec = audio.log_mel_spectrogram(audio_arr, n_mels=n_mels)
        return audio_arr, mel_spec

    @staticmethod
    def preprocess_hf_audio(
        waveform: np.ndarray,
        sampling_rate: int,
        n_mels: int = DEFAULT_N_MELS,
        for_long_form: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Preprocess audio arrays from HuggingFace datasets.

        Processes raw audio waveforms from HuggingFace dataset format, handling
        resampling, normalization, and feature extraction for both short-form
        and long-form transcription tasks.

        Args:
            waveform (np.ndarray): Raw audio waveform array
            sampling_rate (int): Original sampling rate of the waveform
            n_mels (int, optional): Number of mel-spectrogram bins for short-form.
                Defaults to DEFAULT_N_MELS.
            for_long_form (bool, optional): If True, return only waveform for long-form
                transcription. If False, also compute mel-spectrogram. Defaults to False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
                - If for_long_form=True: Preprocessed audio waveform only
                - If for_long_form=False: Tuple of (audio_waveform, mel_spectrogram)

        Example:
            >>> # Short-form preprocessing
            >>> audio_arr, mel_spec = AudioProcessor.preprocess_hf_audio(
            ...     waveform, 22050, for_long_form=False
            ... )
            >>> # Long-form preprocessing
            >>> audio_arr = AudioProcessor.preprocess_hf_audio(
            ...     waveform, 22050, for_long_form=True
            ... )
        """
        if sampling_rate != SAMPLE_RATE:
            waveform = librosa.resample(
                waveform, orig_sr=sampling_rate, target_sr=SAMPLE_RATE
            )

        audio_arr = waveform.astype(np.float32)

        if for_long_form:
            return audio_arr

        audio_arr = audio.pad_or_trim(audio_arr)
        audio_input = audio.log_mel_spectrogram(audio_arr, n_mels=n_mels)
        return audio_arr, audio_input


class TextCleaner:
    """Handles text cleaning operations for different datasets.

    This utility class provides dataset-specific text normalization and cleaning
    operations to ensure consistent transcript processing across different evaluation
    datasets, particularly for specialized corpora like CORAAL.

    Methods:
        clean_coraal_text: Specialized text cleaning for CORAAL dialect corpus
    """

    @staticmethod
    def clean_coraal_text(text: str) -> str:
        """Clean CORAAL dataset text according to corpus conventions.

        Applies CORAAL-specific text normalization including dialect word mapping,
        marker removal, and unintelligible segment handling. This ensures fair
        evaluation by standardizing transcript format.

        Args:
            text (str): Raw CORAAL transcript text with corpus-specific markup

        Returns:
            str: Cleaned transcript text ready for evaluation

        Note:
            CORAAL (Corpus of Regional African American Language) uses specific
            markup conventions for dialectal features, background noise, and
            unintelligible segments that need standardized handling.

        Example:
            >>> raw_text = "We(BR) aksed for [unintelligible] busses"
            >>> clean_text = TextCleaner.clean_coraal_text(raw_text)
            >>> print(clean_text)
            "We asked for  buses"
        """
        text = text.replace("[", "{").replace("]", "}")

        # Relabel CORAAL words
        replacements = {
            "busses": "buses",
            "aks": "ask",
            "aksing": "asking",
            "aksed": "asked",
        }
        words = text.split()
        words = [replacements.get(word, word) for word in words]
        text = " ".join(words)

        # Remove CORAAL flags and markers
        patterns_to_remove = [
            r"(?i)\/unintelligible\/",
            r"(?i)\/inaudible\/",
            r"\/RD(.*?)\/",
            r"\/(\?)\1*\/",
        ]
        for pattern in patterns_to_remove:
            text = re.sub(pattern, "", text)

        # Remove nonlinguistic markers
        markers = [("<", ">"), ("(", ")"), ("{", "}")]
        for start, end in markers:
            text = re.sub(f" ?\\{start}[^{end}]+\\{end}", "", text)

        return text


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders.

    Defines the common interface for all dataset loaders in the evaluation pipeline.
    Each concrete loader implements dataset-specific logic for loading audio files
    and their corresponding transcripts.

    Attributes:
        root_dir (str): Root directory containing the dataset files

    Methods:
        load: Abstract method to load dataset audio files and transcripts
    """

    def __init__(self, root_dir: str):
        """Initialize the dataset loader with root directory.

        Args:
            root_dir (str): Path to the root directory containing dataset files
        """
        self.root_dir = root_dir

    @abstractmethod
    def load(self) -> Tuple[list, list]:
        """Load audio files and transcripts from the dataset.

        Returns:
            Tuple[list, list]: A tuple containing:
                - List of audio file paths or AudioSegment objects
                - List of corresponding transcript strings

        Raises:
            NotImplementedError: If not implemented by concrete subclass
        """
        pass


class LibrispeechLoader(BaseDatasetLoader):
    """Dataset loader for LibriSpeech test sets.

    LibriSpeech is a large corpus of approximately 1000 hours of English speech
    derived from LibriVox audiobooks. This loader handles both 'test-clean' and
    'test-other' evaluation subsets.

    Dataset Structure:
        - Audio files in FLAC format organized by speaker/chapter
        - Transcript files (.txt) contain utterance ID and normalized text
        - Standard format: SPEAKER-CHAPTER-UTTERANCE_ID transcript_text

    Returns:
        Tuple of audio file paths and corresponding transcript texts

    Reference:
        Panayotov, V., et al. "Librispeech: an ASR corpus based on public domain audio books."
    """

    def load(self) -> Tuple[list, list]:
        """Load LibriSpeech audio files and transcripts.

        Recursively searches for transcript files and maps them to corresponding
        FLAC audio files using the LibriSpeech naming convention.

        Returns:
            Tuple[list, list]: A tuple containing:
                - List of audio file paths (FLAC format)
                - List of corresponding transcript strings
        """
        transcript_files = []
        audio_text = {}

        for root, _, files in os.walk(self.root_dir):
            transcript_files.extend(
                os.path.join(root, file) for file in files if file.endswith(".txt")
            )

        for file in sorted(transcript_files):
            with open(file, "r") as f:
                for line in f:
                    parts = line.split(" ")
                    audio_codes = parts[0].split("-")
                    audio_file = os.path.join(
                        self.root_dir,
                        audio_codes[0],
                        audio_codes[1],
                        f"{audio_codes[0]}-{audio_codes[1]}-{audio_codes[2]}.flac",
                    )
                    audio_text[audio_file] = " ".join(parts[1:]).strip()

        return list(audio_text.keys()), list(audio_text.values())


class ArtieBiasCorpusLoader(BaseDatasetLoader):
    """Dataset loader for Artie Bias Corpus.

    The Artie Bias Corpus is designed to evaluate ASR systems for bias and fairness
    across different demographic groups. It contains demographically-annotated speech
    samples to test for systematic performance differences.

    Dataset Structure:
        - TSV file format with audio paths and transcripts
        - Includes demographic metadata for bias analysis
        - Audio files in various formats (typically WAV/MP3)

    Returns:
        Tuple of audio file paths and corresponding transcript texts

    Reference:
        Meyer, J., et al. "Artie Bias Corpus: An Open Dataset for Detecting Demographic
        Bias in Speech Recognition Systems."
    """

    def load(self) -> Tuple[list, list]:
        """Load Artie Bias Corpus audio files and transcripts.

        Parses the TSV metadata file to extract audio file paths and their
        corresponding transcript texts.

        Returns:
            Tuple[list, list]: A tuple containing:
                - List of audio file paths
                - List of corresponding transcript strings
        """
        audio_files, transcript_texts = [], []

        with open(os.path.join(self.root_dir, "artie-bias-corpus.tsv"), "r") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.split("\t")
                audio_files.append(os.path.join(self.root_dir, parts[1].strip()))
                transcript_texts.append(parts[2].strip())

        return audio_files, transcript_texts


class FleursLoader(BaseDatasetLoader):
    """Dataset loader for FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech).

    FLEURS is a multilingual speech dataset spanning 102 languages, designed for
    few-shot learning evaluation. This loader specifically handles the English subset.

    Dataset Structure:
        - TSV file format with audio paths and transcripts
        - Standardized across all language subsets
        - Audio files in MP3/WAV format

    Returns:
        Tuple of audio file paths and corresponding transcript texts

    Reference:
        Conneau, A., et al. "FLEURS: Few-shot Learning Evaluation of Universal
        Representations of Speech."
    """

    def load(self) -> Tuple[list, list]:
        """Load FLEURS dataset audio files and transcripts.

        Parses the test.tsv file to extract audio file paths and transcriptions
        for the English language subset.

        Returns:
            Tuple[list, list]: A tuple containing:
                - List of audio file paths from test directory
                - List of corresponding transcript strings
        """
        with open(f"{self.root_dir}/test.tsv", "r") as f:
            file_text = [line.split("\t")[1:3] for line in f]
            audio_files, transcript_texts = zip(*file_text)
            audio_files = [f"{self.root_dir}/test/{f}" for f in audio_files]
        return list(audio_files), list(transcript_texts)


class VoxPopuliLoader(BaseDatasetLoader):
    """Dataset loader for VoxPopuli dataset.

    VoxPopuli is a large-scale multilingual speech corpus for representation learning
    and speech recognition, extracted from European Parliament event recordings.
    This loader handles the English subset.

    Dataset Structure:
        - TSV file format with utterance IDs and normalized text
        - Audio files derived from parliamentary proceedings
        - Covers formal, political speech domain

    Returns:
        Tuple of audio file paths and corresponding transcript texts

    Reference:
        Wang, C., et al. "VoxPopuli: A Large-Scale Multilingual Speech Corpus for
        Representation Learning, Semi-Supervised Learning and Interpretation."
    """

    def load(self) -> Tuple[list, list]:
        """Load VoxPopuli audio files and transcripts.

        Parses the ASR test TSV file to extract audio file paths and normalized
        transcript texts for English parliamentary speech.

        Returns:
            Tuple[list, list]: A tuple containing:
                - List of audio file paths (WAV format)
                - List of corresponding normalized transcript strings
        """
        with open(f"{self.root_dir}/asr_test.tsv", "r") as f:
            next(f)  # Skip header
            file_text = [line.split("\t")[:2] for line in f]
            audio_files, transcript_texts = zip(*file_text)
            audio_files = [f"{self.root_dir}/test/{f}.wav" for f in audio_files]
        return list(audio_files), list(transcript_texts)


class AMILoader(BaseDatasetLoader):
    """Dataset loader for AMI Meeting Corpus.

    The AMI corpus consists of 100 hours of meeting recordings captured using
    multiple microphones. This loader supports both Individual Headset Microphone (IHM)
    and Single Distant Microphone (SDM) conditions.

    Dataset Structure:
        - Text file with utterance IDs and transcripts
        - Audio files organized by meeting session
        - Multiple microphone configurations available

    Returns:
        Tuple of audio file paths and corresponding transcript texts

    Reference:
        Carletta, J., et al. "The AMI Meeting Corpus: A Pre-announcement."
    """

    def load(self) -> Tuple[list, list]:
        """Load AMI corpus audio files and transcripts.

        Parses the text file to extract utterance IDs and maps them to
        corresponding audio files in the evaluation subset.

        Returns:
            Tuple[list, list]: A tuple containing:
                - List of audio file paths (WAV format)
                - List of corresponding transcript strings
        """
        with open(f"{self.root_dir}/text", "r") as f:
            file_text = [line.split(" ", 1) for line in f]
            audio_files, transcript_texts = zip(*file_text)
            audio_files = [
                f"{self.root_dir}/{f.split('_')[1]}/eval_{f.lower()}.wav"
                for f in audio_files
            ]
        return list(audio_files), list(transcript_texts)


class CORAALLoader(BaseDatasetLoader):
    """Dataset loader for CORAAL (Corpus of Regional African American Language).

    CORAAL is a corpus of African American Language featuring audio-aligned transcripts
    from multiple U.S. regions. This loader handles dialect-specific text normalization
    and corpus markup conventions.

    Dataset Structure:
        - CSV file with segment metadata and transcripts
        - Audio files organized by geographical region
        - Specialized markup for dialectal features

    Returns:
        Tuple of audio file paths and corresponding cleaned transcript texts

    Reference:
        Kendall, T., & Farrington, C. "The Corpus of Regional African American Language."
    """

    def load(self) -> Tuple[list, list]:
        """Load CORAAL audio files and cleaned transcripts.

        Parses the CORAAL transcripts CSV and applies specialized text cleaning
        to handle dialectal markup and corpus conventions.

        Returns:
            Tuple[list, list]: A tuple containing:
                - List of audio file paths (WAV/MP3 format)
                - List of cleaned transcript strings
        """
        audio_files, transcript_texts = [], []
        segments = pd.read_csv(
            f"{self.root_dir}/CORAAL_transcripts.csv", quotechar='"'
        ).values.tolist()

        for segment in segments:
            segment_filename, _, _, _, source, _, _, content, *_ = segment
            sub_folder = os.path.join(self.root_dir, "CORAAL_audio", source.lower())
            audio_file = os.path.join(sub_folder, segment_filename)

            if not os.path.exists(audio_file):
                audio_file = audio_file.replace(".wav", ".mp3")

            audio_files.append(audio_file)
            transcript_texts.append(TextCleaner.clean_coraal_text(content))

        return audio_files, transcript_texts


class Chime6Loader(BaseDatasetLoader):
    """Dataset loader for CHiME-6 Challenge dataset.

    CHiME-6 focuses on distant multi-microphone conversational speech recognition
    in everyday home environments. The challenge addresses far-field speech recognition
    with multiple speakers and environmental noise.

    Dataset Structure:
        - JSON transcript files with segment metadata
        - Audio segments with precise timing information
        - Multi-microphone array recordings

    Returns:
        Tuple of audio file paths and corresponding transcript texts

    Reference:
        Watanabe, S., et al. "CHiME-6 Challenge: Tackling Multispeaker Speech Recognition
        for Unsegmented Recordings."
    """

    def load(self) -> Tuple[list, list]:
        """Load CHiME-6 audio segments and transcripts.

        Parses JSON transcript files to extract audio segment paths and their
        corresponding word-level transcriptions.

        Returns:
            Tuple[list, list]: A tuple containing:
                - List of audio segment file paths
                - List of corresponding transcript strings
        """
        audio_files, transcript_texts = [], []

        for p in glob.glob(f"{self.root_dir}/transcripts/*.json"):
            with open(p, "r") as f:
                data = json.load(f)
                audio_files.extend(
                    [
                        os.path.join(
                            self.root_dir, "segments", segment_dict["audio_seg_file"]
                        )
                        for segment_dict in data
                    ]
                )
                transcript_texts.extend(
                    [segment_dict["words"] for segment_dict in data]
                )

        return audio_files, transcript_texts


class WSJLoader(BaseDatasetLoader):
    """Dataset loader for Wall Street Journal (WSJ) corpus.

    The WSJ corpus contains read speech from the Wall Street Journal newspaper,
    representing formal, news-domain speech. This loader handles the Kaldi-format
    data organization with separate text and audio script files.

    Dataset Structure:
        - Kaldi format with wav.scp and text files
        - Shell commands for audio extraction
        - Formal, read speech content

    Returns:
        Tuple of audio extraction commands and corresponding transcript texts

    Reference:
        Paul, D. B., & Baker, J. M. "The design for the Wall Street Journal-based CSR corpus."
    """

    def load(self) -> Tuple[list, list]:
        """Load WSJ audio extraction commands and transcripts.

        Parses Kaldi-format text and wav.scp files to create mappings between
        utterance IDs, audio extraction commands, and transcript texts.

        Returns:
            Tuple[list, list]: A tuple containing:
                - List of shell commands for audio extraction
                - List of corresponding transcript strings
        """
        audio_files, transcript_files = [], []

        for direc in glob.glob(f"{self.root_dir}/test_eval*"):
            with open(f"{direc}/text") as f:
                id_2_text = {
                    line.strip()
                    .split(" ")[0]: line.strip()
                    .split(" ", maxsplit=1)[-1]
                    .strip()
                    for line in f
                }

            with open(f"{direc}/wav.scp") as f:
                for line in f:
                    audio_file = line.strip().split(" ", maxsplit=1)[-1].split(" |")[0]
                    utter_id = line.strip().split(" ")[0]
                    transcript_text = id_2_text[utter_id]
                    audio_files.append(audio_file)
                    transcript_files.append(transcript_text)

        return audio_files, transcript_files


class CallHomeLoader(BaseDatasetLoader):
    """Dataset loader for CallHome English corpus.

    CallHome consists of conversational telephone speech between family members
    and friends, representing informal, spontaneous speech patterns. This loader
    handles the HUB5 evaluation format with multi-channel audio.

    Dataset Structure:
        - HUB5 STM format with timing and channel information
        - SPHERE audio files requiring channel extraction
        - Conversational telephone speech

    Returns:
        Tuple of AudioSegment objects with timing and corresponding transcript texts

    Reference:
        Canavan, A., et al. "CALLHOME American English Speech."
    """

    def load(self) -> Tuple[list, list]:
        """Load CallHome audio segments and transcripts.

        Delegates to the shared HUB5 data loading logic with English prefix
        to extract CallHome-specific segments.

        Returns:
            Tuple[list, list]: A tuple containing:
                - List of AudioSegment tuples (wav_file, start_time, end_time)
                - List of corresponding transcript strings
        """
        return self._load_hub5_data("en")


class SwitchBoardLoader(BaseDatasetLoader):
    """Dataset loader for Switchboard corpus.

    Switchboard contains conversational telephone speech between strangers discussing
    predetermined topics. Like CallHome, it uses the HUB5 evaluation format but
    represents more structured conversational speech.

    Dataset Structure:
        - HUB5 STM format with timing and channel information
        - SPHERE audio files requiring channel extraction
        - Topic-guided conversational speech

    Returns:
        Tuple of AudioSegment objects with timing and corresponding transcript texts

    Reference:
        Godfrey, J. J., et al. "SWITCHBOARD: telephone speech corpus for research and development."
    """

    def load(self) -> Tuple[list, list]:
        """Load Switchboard audio segments and transcripts.

        Delegates to the shared HUB5 data loading logic with Switchboard prefix
        to extract Switchboard-specific segments.

        Returns:
            Tuple[list, list]: A tuple containing:
                - List of AudioSegment tuples (wav_file, start_time, end_time)
                - List of corresponding transcript strings
        """
        return self._load_hub5_data("sw")

    def _load_hub5_data(self, prefix: str) -> Tuple[list, list]:
        """Common logic for CallHome and SwitchBoard datasets.

        Parses HUB5 STM format files to extract audio segments with precise timing
        information. Handles channel separation using Sox audio processing.

        Args:
            prefix (str): Dataset prefix ("en" for CallHome, "sw" for Switchboard)

        Returns:
            Tuple[list, list]: A tuple containing:
                - List of AudioSegment tuples (wav_file, start_time, end_time)
                - List of corresponding transcript strings

        Note:
            Requires Sox audio processing tool for SPHERE format conversion
            and channel extraction.
        """
        audio_files, transcript_files = [], []

        stm_path = f"{self.root_dir}/2000_hub5_eng_eval_tr/reference/hub5e00.english.000405.stm"
        with open(stm_path, "r") as f:
            for line in f:
                if line.startswith(";;") or not line.startswith(prefix):
                    continue

                parts = line.split(" ")
                audio_file = f"{self.root_dir}/hub5e_00/english/{parts[0]}.sph"
                channel = parts[1]

                wav_file = f"{audio_file.split('.')[0]}_{channel}.wav"
                if not os.path.exists(wav_file):
                    remix_channel = "1" if channel == "A" else "2"
                    subprocess.run(
                        ["sox", audio_file, wav_file, "remix", remix_channel],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )

                transcript_text = re.split(r"<[^>]+>", line)[-1].strip()
                start_time = float(parts[3])

                # Find end_time from available parts
                end_time = None
                for i in [4, 5, 6]:
                    if i < len(parts) and parts[i]:
                        end_time = float(parts[i])
                        break

                audio_files.append((wav_file, start_time, end_time))
                transcript_files.append(transcript_text)

        return audio_files, transcript_files


class Kincaid46Loader(BaseDatasetLoader):
    """Dataset loader for Kincaid46 corpus.

    Kincaid46 is a specialized evaluation dataset containing 46 utterances designed
    to test specific aspects of speech recognition performance. The dataset focuses
    on carefully selected linguistic content for controlled evaluation.

    Dataset Structure:
        - CSV file format with metadata and transcripts
        - M4A audio files with zero-padded naming convention
        - Compact size for targeted evaluation scenarios

    Returns:
        Tuple of audio file paths and corresponding transcript texts

    Note:
        This is a smaller, specialized corpus primarily used for specific
        evaluation scenarios or as a subset for quick testing.
    """

    def load(self) -> Tuple[list, list]:
        """Load Kincaid46 audio files and transcripts.

        Parses the CSV metadata file and maps entries to corresponding M4A audio
        files using a zero-padded naming convention.

        Returns:
            Tuple[list, list]: A tuple containing:
                - List of M4A audio file paths
                - List of corresponding transcript strings from CSV column 5
        """
        audio_files, transcript_texts = [], []

        with open(f"{self.root_dir}/text.csv", "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                audio_file = os.path.join(self.root_dir, "audio", f"{(i - 1):02}.m4a")
                transcript_texts.append(row[5])
                audio_files.append(audio_file)

        return audio_files, transcript_texts


class CORAALLongLoader(BaseDatasetLoader):
    """Dataset loader for CORAAL long-form transcription variant.

    This loader handles the long-form version of the CORAAL corpus, designed for
    extended speech recognition evaluation. Unlike the segmented CORAAL corpus,
    this version provides longer continuous audio segments.

    Dataset Structure:
        - JSONL format with audio paths and full transcripts
        - Longer audio segments for long-form evaluation
        - Pre-cleaned transcript text (no additional markup processing needed)

    Returns:
        Tuple of audio file paths and corresponding transcript texts

    Reference:
        Extended version of Kendall, T., & Farrington, C. "The Corpus of Regional
        African American Language" adapted for long-form transcription tasks.
    """

    def load(self) -> Tuple[list, list]:
        """Load CORAAL long-form audio files and transcripts.

        Parses the JSONL file to extract audio file paths and their corresponding
        full transcript texts for long-form evaluation.

        Returns:
            Tuple[list, list]: A tuple containing:
                - List of audio file paths
                - List of complete transcript strings
        """
        audio_files, transcript_texts = [], []

        with open(f"{self.root_dir}/coraal_transcripts.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                audio_files.append(data["audio"])
                transcript_texts.append(data["text"])

        return audio_files, transcript_texts


class DatasetFactory:
    """Factory pattern implementation for creating dataset loaders.

    This factory class provides a centralized way to instantiate appropriate dataset
    loaders based on evaluation set names. It maintains a registry of all supported
    datasets and their corresponding loader classes.

    The factory pattern allows for easy extension of new datasets without modifying
    existing code, promoting the open-closed principle.

    Attributes:
        _loaders (Dict[str, Type[BaseDatasetLoader]]): Registry mapping dataset names
            to their corresponding loader classes

    Methods:
        create_loader: Factory method to instantiate appropriate dataset loader

    Supported Datasets:
        - LibriSpeech (clean/other): Large audiobook corpus
        - Artie Bias Corpus: Bias evaluation dataset
        - FLEURS: Multilingual few-shot learning corpus
        - VoxPopuli: Parliamentary speech corpus
        - AMI: Meeting corpus (IHM/SDM)
        - CORAAL: African American Language corpus (standard/long)
        - CHiME-6: Multi-microphone challenge corpus
        - WSJ: Wall Street Journal read speech
        - CallHome: Conversational telephone speech
        - Switchboard: Topic-guided conversation corpus
        - Kincaid46: Specialized evaluation corpus

    Example:
        >>> factory = DatasetFactory()
        >>> loader = factory.create_loader("librispeech_clean", "/path/to/data")
        >>> audio_files, transcripts = loader.load()
    """

    _loaders = {
        "librispeech_clean": LibrispeechLoader,
        "librispeech_other": LibrispeechLoader,
        "artie_bias_corpus": ArtieBiasCorpusLoader,
        "fleurs": FleursLoader,
        "voxpopuli": VoxPopuliLoader,
        "ami_ihm": AMILoader,
        "ami_sdm": AMILoader,
        "coraal": CORAALLoader,
        "chime6": Chime6Loader,
        "wsj": WSJLoader,
        "callhome": CallHomeLoader,
        "switchboard": SwitchBoardLoader,
        "kincaid46": Kincaid46Loader,
        "coraal_long": CORAALLongLoader,
    }

    @classmethod
    def create_loader(cls, eval_set: str, root_dir: str) -> BaseDatasetLoader:
        """Create appropriate dataset loader for the specified evaluation set.

        Factory method that instantiates the correct dataset loader class based on
        the evaluation set name. Provides type safety and centralized loader management.

        Args:
            eval_set (str): Name of the evaluation dataset (must be in supported list)
            root_dir (str): Root directory path containing the dataset files

        Returns:
            BaseDatasetLoader: Instantiated dataset loader for the specified eval_set

        Raises:
            ValueError: If eval_set is not in the supported datasets registry

        Example:
            >>> loader = DatasetFactory.create_loader("librispeech_clean", "/data/libri")
            >>> audio_files, transcripts = loader.load()
            >>> print(f"Loaded {len(audio_files)} audio files")
        """
        if eval_set not in cls._loaders:
            raise ValueError(f"Unknown eval_set: {eval_set}")
        return cls._loaders[eval_set](root_dir)


# Legacy classes for backwards compatibility
Librispeech = LibrispeechLoader
ArtieBiasCorpus = ArtieBiasCorpusLoader
Fleurs = FleursLoader
VoxPopuli = VoxPopuliLoader
AMI = AMILoader
CORAAL = CORAALLoader
chime6 = Chime6Loader
WSJ = WSJLoader
CallHome = CallHomeLoader
SwitchBoard = SwitchBoardLoader
Kincaid46 = Kincaid46Loader
CORAAL_long = CORAALLongLoader


class EvalDataset(Dataset):
    """PyTorch Dataset for ASR evaluation across multiple datasets and tasks.

    This dataset class provides a unified interface for loading and preprocessing
    evaluation data for both short-form and long-form speech recognition tasks.
    It handles the complexity of different dataset formats, audio preprocessing,
    and HuggingFace integration.

    The dataset supports two main evaluation paradigms:
    1. Short-form transcription: Fixed-length audio segments with mel-spectrograms
    2. Long-form transcription: Variable-length audio for extended speech

    Attributes:
        eval_set (str): Name of the evaluation dataset
        task (str): Transcription task type ("eng_transcribe" or "long_form_transcribe")
        n_mels (int): Number of mel-spectrogram bins for short-form tasks
        audio_processor (AudioProcessor): Audio preprocessing utilities
        dataset: Loaded dataset (HuggingFace or custom loader)
        audio_files (list): Audio file paths (for custom datasets)
        transcript_texts (list): Transcript texts (for custom datasets)

    Supported Datasets:
        Short-form: LibriSpeech, TEDLIUM, WSJ, CallHome, Switchboard, Common Voice,
                   Artie Bias Corpus, CORAAL, CHiME-6, AMI, VoxPopuli, FLEURS
        Long-form: TEDLIUM, Meanwhile, Rev16, Earnings21, Earnings22, CORAAL, Kincaid46

    Example:
        >>> # Short-form evaluation
        >>> dataset = EvalDataset(
        ...     task="eng_transcribe",
        ...     eval_set="librispeech_clean",
        ...     eval_dir="data/eval"
        ... )
        >>> dataloader = DataLoader(dataset, batch_size=32)
        >>>
        >>> # Long-form evaluation
        >>> dataset = EvalDataset(
        ...     task="long_form_transcribe",
        ...     eval_set="tedlium",
        ...     eval_dir="data/eval"
        ... )
    """

    def __init__(
        self,
        task: Literal["eng_transcribe", "long_form_transcribe"],
        eval_set: Literal[
            "librispeech_clean",
            "librispeech_other",
            "tedlium",
            "wsj",
            "callhome",
            "switchboard",
            "common_voice",
            "artie_bias_corpus",
            "coraal",
            "chime6",
            "ami_ihm",
            "ami_sdm",
            "voxpopuli",
            "fleurs",
            "meanwhile",
            "kincaid46",
            "rev16",
            "earnings21",
            "earnings22",
        ],
        hf_token: Optional[str] = None,
        eval_dir: str = "data/eval",
        n_mels: int = DEFAULT_N_MELS,
    ):
        """Initialize the evaluation dataset.

        Sets up the appropriate dataset loader based on the evaluation set and task type.
        Handles both HuggingFace datasets and custom dataset loaders with automatic
        data downloading if needed.

        Args:
            task (Literal): Type of transcription task - either "eng_transcribe" for
                short-form or "long_form_transcribe" for extended audio
            eval_set (Literal): Name of the evaluation dataset to load
            hf_token (Optional[str], optional): HuggingFace authentication token for
                private datasets. Defaults to None.
            eval_dir (str, optional): Directory for storing evaluation datasets.
                Defaults to "data/eval".
            n_mels (int, optional): Number of mel-spectrogram bins for short-form tasks.
                Defaults to DEFAULT_N_MELS.

        Raises:
            ValueError: If eval_set is not supported for the specified task
            FileNotFoundError: If required dataset files are not found and cannot be downloaded

        Example:
            >>> dataset = EvalDataset(
            ...     task="eng_transcribe",
            ...     eval_set="librispeech_clean",
            ...     eval_dir="/data/eval",
            ...     n_mels=80
            ... )
        """
        self.eval_set = eval_set
        self.task = task
        self.n_mels = n_mels
        self.audio_processor = AudioProcessor()

        if eval_set in HF_DATASETS:
            self._init_hf_dataset(eval_set, eval_dir, hf_token, task)
        else:
            self._init_custom_dataset(eval_set, eval_dir, task)

    def _init_hf_dataset(
        self, eval_set: str, eval_dir: str, hf_token: Optional[str], task: str
    ):
        """Initialize HuggingFace datasets."""
        dataset_configs = {
            "fleurs": {
                "path": "google/fleurs",
                "name": "en_us",
                "cache_subdir": "google___fleurs/en_us",
            },
            "voxpopuli": {
                "path": "facebook/voxpopuli",
                "name": "en",
                "cache_subdir": "facebook___voxpopuli",
            },
            "common_voice": {
                "path": "mozilla-foundation/common_voice_5_1",
                "name": "en",
                "cache_subdir": "mozilla-foundation___common_voice_5_1",
            },
            "tedlium": {
                "path": (
                    "distil-whisper/tedlium-long-form"
                    if task == "long_form_transcribe"
                    else None
                )
            },
            "meanwhile": {"path": "distil-whisper/meanwhile"},
            "rev16": {"path": "distil-whisper/rev16", "name": "whisper_subset"},
            "earnings21": {"path": "distil-whisper/earnings21", "name": "full"},
            "earnings22": {"path": "distil-whisper/earnings22", "name": "full"},
        }

        config = dataset_configs[eval_set]

        if eval_set == "tedlium" and task == "eng_transcribe":
            # Use TEDLIUM from torchaudio for eng_transcribe
            if not os.path.exists(f"{eval_dir}/TEDLIUM_release-3"):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)
            self.dataset = TEDLIUM(root=eval_dir, release="release3", subset="test")
        else:
            # Check if dataset exists
            cache_subdir = config.get(
                "cache_subdir", config["path"].replace("/", "___")
            )
            if not os.path.exists(f"{eval_dir}/{cache_subdir}"):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir, hf_token=hf_token)

            # Load dataset
            load_kwargs = {
                "path": config["path"],
                "split": "test",
                "cache_dir": eval_dir,
                "trust_remote_code": True,
                "num_proc": 15,
                "save_infos": True,
            }

            if "name" in config:
                load_kwargs["name"] = config["name"]
            if hf_token:
                load_kwargs["token"] = hf_token

            self.dataset = load_dataset(**load_kwargs)

    def _init_custom_dataset(self, eval_set: str, eval_dir: str, task: str):
        """Initialize custom datasets."""
        root_dirs = {
            "librispeech_clean": f"{eval_dir}/librispeech_test_clean",
            "librispeech_other": f"{eval_dir}/librispeech_test_other",
            "artie_bias_corpus": f"{eval_dir}/artie-bias-corpus",
            "ami_ihm": f"{eval_dir}/ami/ihm",
            "ami_sdm": f"{eval_dir}/ami/sdm",
            "coraal": (
                f"{eval_dir}/coraal_long"
                if task == "long_form_transcribe"
                else f"{eval_dir}/coraal"
            ),
            "chime6": f"{eval_dir}/chime6",
            "wsj": f"{eval_dir}/kaldi/egs/wsj/s5/data",
            "callhome": eval_dir,
            "switchboard": eval_dir,
            "kincaid46": f"{eval_dir}/kincaid46",
        }

        root_dir = root_dirs[eval_set]

        # Download dataset if needed
        if not os.path.exists(root_dir) and eval_set not in [
            "wsj",
            "callhome",
            "switchboard",
        ]:
            get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

        # Create appropriate loader
        loader_eval_set = (
            "coraal_long"
            if task == "long_form_transcribe" and eval_set == "coraal"
            else eval_set
        )
        self.dataset = DatasetFactory.create_loader(loader_eval_set, root_dir)

        # Load data for non-HF datasets
        audio_files, transcript_texts = self.dataset.load()
        self.audio_files = audio_files
        self.transcript_texts = transcript_texts

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            int: Total number of audio samples available for evaluation

        Note:
            For HuggingFace datasets, returns the length of the loaded dataset.
            For custom datasets, returns the length of the audio files list.
        """
        if self.eval_set in HF_DATASETS:
            return len(self.dataset)
        return len(self.audio_files)

    def __getitem__(self, index):
        """Get a single sample from the dataset.

        Dispatches to appropriate method based on the transcription task type.
        Returns preprocessed audio and transcript for model evaluation.

        Args:
            index (int): Index of the sample to retrieve

        Returns:
            Tuple: Task-specific tuple containing audio data and transcript
                - For short-form: (audio_fp, audio_arr, audio_input, text_y)
                - For long-form: (audio_fp, audio_arr, audio_input, text_y)

        Example:
            >>> dataset = EvalDataset(task="eng_transcribe", eval_set="librispeech_clean")
            >>> audio_fp, audio_arr, mel_spec, transcript = dataset[0]
            >>> print(f"Audio shape: {audio_arr.shape}, Transcript: {transcript}")
        """
        if self.task == "eng_transcribe":
            return self._get_short_form_item(index)
        else:
            return self._get_long_form_item(index)

    def _get_short_form_item(self, index):
        """Get item for short-form transcription.

        Retrieves and preprocesses audio for fixed-length transcription tasks.
        Handles both HuggingFace and custom dataset formats.

        Args:
            index (int): Index of the sample to retrieve

        Returns:
            Tuple[str, np.ndarray, torch.Tensor, str]: A tuple containing:
                - audio_fp: Audio file path (empty string for some datasets)
                - audio_arr: Raw audio waveform array
                - audio_input: Mel-spectrogram tensor for model input
                - text_y: Ground truth transcript string
        """
        if self.eval_set in HF_DATASETS:
            return self._get_hf_short_form_item(index)
        else:
            return self._get_custom_short_form_item(index)

    def _get_hf_short_form_item(self, index):
        """Get HuggingFace dataset item for short-form transcription.

        Processes audio and text from HuggingFace dataset format, handling
        dataset-specific text field mappings and audio preprocessing.

        Args:
            index (int): Index of the sample to retrieve

        Returns:
            Tuple[str, np.ndarray, torch.Tensor, str]: A tuple containing:
                - audio_fp: Audio file path from HuggingFace metadata
                - audio_arr: Preprocessed audio waveform
                - audio_input: Mel-spectrogram features
                - text_y: Ground truth transcript from appropriate field
        """
        item = self.dataset[index]

        text_field_map = {
            "tedlium": "text",
            "common_voice": "sentence",
            "fleurs": "transcription",
            "voxpopuli": "normalized_text",
        }

        waveform = item["audio"]["array"]
        audio_fp = item["audio"]["path"]
        sampling_rate = item["audio"]["sampling_rate"]
        text_y = item[text_field_map[self.eval_set]]

        audio_arr, audio_input = self.audio_processor.preprocess_hf_audio(
            waveform, sampling_rate, self.n_mels, for_long_form=False
        )

        return audio_fp, audio_arr, audio_input, text_y

    def _get_custom_short_form_item(self, index):
        """Get custom dataset item for short-form transcription.

        Handles dataset-specific loading and preprocessing for non-HuggingFace
        datasets including special cases like WSJ command execution and
        segmented audio from CallHome/Switchboard.

        Args:
            index (int): Index of the sample to retrieve

        Returns:
            Tuple[str, np.ndarray, torch.Tensor, str]: A tuple containing:
                - audio_fp: Audio file path (or empty for WSJ)
                - audio_arr: Preprocessed audio waveform
                - audio_input: Mel-spectrogram features
                - text_y: Ground truth transcript

        Note:
            WSJ dataset requires shell command execution for audio extraction.
            CallHome/Switchboard use time-segmented audio with precise timing.
        """
        audio_fp = ""
        text_y = self.transcript_texts[index]

        if self.eval_set == "wsj":
            result = subprocess.run(
                self.audio_files[index],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            audio_bytes = io.BytesIO(result.stdout)
            audio_arr, _ = torchaudio.load(audio_bytes)
            audio_arr = audio_arr.squeeze(0)
            audio_arr = audio.pad_or_trim(audio_arr).float()
            audio_input = audio.log_mel_spectrogram(audio_arr, n_mels=self.n_mels)
        elif self.eval_set in ["callhome", "switchboard"]:
            audio_fp, start_time, end_time = self.audio_files[index]
            audio_arr, audio_input = self.audio_processor.load_and_preprocess_audio(
                audio_fp,
                sr=SAMPLE_RATE,
                start_time=start_time,
                end_time=end_time,
                n_mels=self.n_mels,
            )
        else:
            audio_fp = self.audio_files[index]
            audio_arr, audio_input = self.audio_processor.load_and_preprocess_audio(
                audio_fp, n_mels=self.n_mels
            )

        return audio_fp, audio_arr, audio_input, text_y

    def _get_long_form_item(self, index):
        """Get item for long-form transcription.

        Retrieves and preprocesses audio for variable-length transcription tasks.
        Does not apply padding or compute mel-spectrograms for long-form evaluation.

        Args:
            index (int): Index of the sample to retrieve

        Returns:
            Tuple[str, np.ndarray, str, str]: A tuple containing:
                - audio_fp: Audio file path (empty for most datasets)
                - audio_arr: Full-length audio waveform
                - audio_input: Empty string (no mel-spectrogram for long-form)
                - text_y: Ground truth transcript string
        """
        if self.eval_set in HF_DATASETS:
            return self._get_hf_long_form_item(index)
        else:
            return self._get_custom_long_form_item(index)

    def _get_hf_long_form_item(self, index):
        """Get HuggingFace dataset item for long-form transcription.

        Processes long-form audio from HuggingFace datasets without padding
        or mel-spectrogram computation. Handles dataset-specific text fields.

        Args:
            index (int): Index of the sample to retrieve

        Returns:
            Tuple[str, np.ndarray, str, str]: A tuple containing:
                - Empty string for audio file path
                - audio_arr: Full-length preprocessed audio waveform
                - Empty string for audio input
                - text_y: Ground truth transcript from appropriate field
        """
        item = self.dataset[index]

        text_field_map = {
            "tedlium": "text",
            "meanwhile": "text",
            "rev16": "transcription",
            "earnings21": "transcription",
            "earnings22": "transcription",
        }

        waveform = item["audio"]["array"]
        sampling_rate = item["audio"]["sampling_rate"]
        text_y = item[text_field_map[self.eval_set]]

        audio_arr = self.audio_processor.preprocess_hf_audio(
            waveform, sampling_rate, for_long_form=True
        )

        return "", audio_arr, "", text_y

    def _get_custom_long_form_item(self, index):
        """Get custom dataset item for long-form transcription.

        Loads and preprocesses audio from custom dataset formats for
        long-form evaluation without padding or feature extraction.

        Args:
            index (int): Index of the sample to retrieve

        Returns:
            Tuple[str, np.ndarray, str, str]: A tuple containing:
                - Empty string for audio file path
                - audio_arr: Full-length preprocessed audio waveform
                - Empty string for audio input
                - text_y: Ground truth transcript
        """
        waveform, sampling_rate = torchaudio.load(self.audio_files[index])
        waveform = waveform.squeeze(0).cpu().numpy()
        text_y = self.transcript_texts[index]

        audio_arr = self.audio_processor.preprocess_hf_audio(
            waveform, sampling_rate, for_long_form=True
        )

        return "", audio_arr, "", text_y

    # Keep original method for backwards compatibility
    def preprocess_audio(
        self, audio_file, sr=SAMPLE_RATE, start_time=None, end_time=None
    ):
        """Legacy method for backwards compatibility.

        Delegates to AudioProcessor for audio preprocessing. Maintained for
        compatibility with existing code that may call this method directly.

        Args:
            audio_file (str): Path to audio file
            sr (int, optional): Target sampling rate. Defaults to SAMPLE_RATE.
            start_time (Optional[float], optional): Start time for segmentation.
            end_time (Optional[float], optional): End time for segmentation.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Audio waveform and mel-spectrogram

        Deprecated:
            Use AudioProcessor.load_and_preprocess_audio() directly instead.
        """
        return self.audio_processor.load_and_preprocess_audio(
            audio_file, sr, start_time, end_time, self.n_mels
        )


class WandBLogger:
    """Handles Weights & Biases logging for ASR evaluation experiments.

    This utility class provides standardized WandB integration for tracking
    ASR evaluation experiments. It handles run initialization, model metadata
    extraction, and logging configuration for both standalone evaluations
    and training-integrated evaluations.

    Key Features:
        - Automatic model type and size detection from checkpoint paths
        - Run ID persistence for experiment continuity
        - Standardized project and entity configuration
        - Evaluation table creation for detailed results logging

    Methods:
        init_wandb: Initialize WandB run with experiment configuration
        _get_model_info: Extract model metadata from checkpoint paths

    Example:
        >>> table = WandBLogger.init_wandb(
        ...     ckpt="models/whisper/large/model.pt",
        ...     eval_set="librispeech_clean",
        ...     wandb_log_dir="logs/wandb"
        ... )
        >>> # Run evaluation and log results to table
    """

    @staticmethod
    def init_wandb(
        ckpt: str,
        eval_set: str,
        train_exp_name: Optional[str] = None,
        train_run_id: Optional[str] = None,
        run_id_dir: Optional[str] = None,
        wandb_log_dir: str = "wandb",
    ):
        """Initialize WandB run for evaluation experiment tracking.

        Sets up a WandB run with appropriate configuration for ASR evaluation,
        including model metadata, experiment naming, and run ID management
        for experiment continuity.

        Args:
            ckpt (str): Path to model checkpoint file
            eval_set (str): Name of evaluation dataset
            train_exp_name (Optional[str], optional): Training experiment name for
                linked evaluation runs. Defaults to None.
            train_run_id (Optional[str], optional): Training run ID for linking
                evaluation to training. Defaults to None.
            run_id_dir (Optional[str], optional): Directory for storing persistent
                run IDs. Defaults to None.
            wandb_log_dir (str, optional): Directory for WandB logs.
                Defaults to "wandb".

        Returns:
            wandb.Table: Initialized WandB table for logging evaluation results
                with columns: eval_set, audio, prediction, target, subs, dels, ins, wer

        Example:
            >>> # Standalone evaluation
            >>> table = WandBLogger.init_wandb(
            ...     ckpt="models/whisper_large.pt",
            ...     eval_set="librispeech_clean"
            ... )
            >>>
            >>> # Training-linked evaluation
            >>> table = WandBLogger.init_wandb(
            ...     ckpt="checkpoints/epoch_10.pt",
            ...     eval_set="tedlium",
            ...     train_exp_name="whisper_training",
            ...     train_run_id="abc123",
            ...     run_id_dir="run_ids/"
            ... )

        Note:
            Run IDs are persisted to disk for training experiments to allow
            resumable evaluation tracking across multiple evaluation runs.
        """
        if train_exp_name is not None:
            run_id_file = f"{run_id_dir}/{train_exp_name}_eval.txt"
            if not os.path.exists(run_id_file):
                run_id = wandb.util.generate_id()
                with open(run_id_file, "w") as f:
                    f.write(run_id)
            else:
                with open(run_id_file, "r") as f:
                    run_id = f.read().strip()
        else:
            run_id = wandb.util.generate_id()

        # Determine model info
        model_info = WandBLogger._get_model_info(ckpt)
        exp_name = (
            f"{eval_set}_eval"
            if model_info["type"] == "whisper"
            else f"ow_{eval_set}_eval"
        )

        config = {
            "ckpt": "/".join(ckpt.split("/")[-2:]),
            "model": model_info["type"],
            "model_size": model_info["size"],
        }
        if train_run_id is not None:
            config["train_run_id"] = train_run_id

        wandb.init(
            id=run_id,
            resume="allow",
            project="olmoasr",
            entity="dogml",
            job_type="evals",
            name=exp_name if train_exp_name is None else train_exp_name,
            dir=wandb_log_dir,
            config=config,
            tags=["eval", eval_set, model_info["type"], model_info["size"]],
        )

        return wandb.Table(
            columns=[
                "eval_set",
                "audio",
                "prediction",
                "target",
                "subs",
                "dels",
                "ins",
                "wer",
            ]
        )

    @staticmethod
    def _get_model_info(ckpt: str) -> Dict[str, str]:
        """Extract model type and size from checkpoint path.

        Analyzes checkpoint file path to automatically determine model type
        (whisper variants) and size for consistent experiment tagging and
        organization in WandB.

        Args:
            ckpt (str): Path to model checkpoint file

        Returns:
            Dict[str, str]: Dictionary containing:
                - "type": Model type ("whisper", "open-whisper", "yodas", "owsm")
                - "size": Model size ("tiny", "small", "base", "medium", "large", "unknown")

        Example:
            >>> info = WandBLogger._get_model_info("models/ow_ckpts/large/model.pt")
            >>> print(info)
            {"type": "open-whisper", "size": "large"}
            >>>
            >>> info = WandBLogger._get_model_info("whisper_ckpts/base.pt")
            >>> print(info)
            {"type": "whisper", "size": "base"}

        Note:
            Model type detection is based on directory structure conventions:
            - ow_ckpts/ -> "open-whisper"
            - yodas/ -> "yodas"
            - owsm/ -> "owsm"
            - other -> "whisper"
        """
        path_parts = ckpt.split("/")

        if len(path_parts) >= 3:
            model_dir = path_parts[-3]
            if model_dir == "ow_ckpts":
                model_type = "open-whisper"
            elif model_dir == "yodas":
                model_type = "yodas"
            elif model_dir == "owsm":
                model_type = "owsm"
            else:
                model_type = "whisper"
        else:
            model_type = "whisper"

        # Extract model size
        model_sizes = ["tiny", "small", "base", "medium", "large"]
        model_size = next((size for size in model_sizes if size in ckpt), "unknown")

        return {"type": model_type, "size": model_size}


def short_form_eval(
    batch_size: int,
    num_workers: int,
    ckpt: str,
    eval_set: Literal[
        "librispeech_clean",
        "librispeech_other",
        "artie_bias_corpus",
        "fleurs",
        "tedlium",
        "voxpopuli",
        "common_voice",
        "ami_ihm",
        "ami_sdm",
        "coraal",
        "chime6",
        "wsj",
        "callhome",
        "switchboard",
    ],
    log_dir: str,
    n_mels: int = DEFAULT_N_MELS,
    current_step: Optional[int] = None,
    train_exp_name: Optional[str] = None,
    train_run_id: Optional[str] = None,
    wandb_log: bool = False,
    wandb_log_dir: Optional[str] = None,
    run_id_dir: Optional[str] = None,
    eval_dir: str = "data/eval",
    hf_token: Optional[str] = None,
    cuda: bool = True,
    bootstrap: bool = False,
):
    """Evaluate ASR model performance on short-form transcription tasks.

    This is the main evaluation function for fixed-length audio segments (typically
    30 seconds or less). It supports multiple evaluation datasets, automatic metric
    computation, and optional experiment tracking via Weights & Biases.

    The function handles the complete evaluation pipeline:
    1. Checkpoint preparation and model loading
    2. Dataset initialization and preprocessing
    3. Batch inference with progress tracking
    4. Metric computation (WER, substitutions, insertions, deletions)
    5. Results logging (files, WandB, console output)
    6. Optional bootstrap sampling for confidence intervals

    Args:
        batch_size (int): Number of audio samples per batch for inference
        num_workers (int): Number of worker processes for data loading
        ckpt (str): Path to model checkpoint file
        eval_set (Literal): Name of evaluation dataset (see supported datasets below)
        log_dir (str): Directory for saving evaluation results and logs
        n_mels (int, optional): Number of mel-spectrogram bins. Defaults to DEFAULT_N_MELS.
        current_step (Optional[int], optional): Training step number for linked evaluation.
            Defaults to None.
        train_exp_name (Optional[str], optional): Training experiment name for organization.
            Defaults to None.
        train_run_id (Optional[str], optional): Training run ID for linking to training.
            Defaults to None.
        wandb_log (bool, optional): Enable Weights & Biases logging. Defaults to False.
        wandb_log_dir (Optional[str], optional): Directory for WandB logs. Defaults to None.
        run_id_dir (Optional[str], optional): Directory for persistent run IDs. Defaults to None.
        eval_dir (str, optional): Root directory for evaluation datasets. Defaults to "data/eval".
        hf_token (Optional[str], optional): HuggingFace token for private datasets. Defaults to None.
        cuda (bool, optional): Use CUDA acceleration if available. Defaults to True.
        bootstrap (bool, optional): Enable bootstrap sampling for confidence intervals.
            Defaults to False.

    Supported Datasets:
        - librispeech_clean: LibriSpeech test-clean (clean read speech)
        - librispeech_other: LibriSpeech test-other (noisy read speech)
        - artie_bias_corpus: Demographic bias evaluation corpus
        - fleurs: Multilingual evaluation corpus (English subset)
        - tedlium: TED talk corpus
        - voxpopuli: Parliamentary speech corpus
        - common_voice: Mozilla Common Voice corpus
        - ami_ihm: AMI meeting corpus (individual headset mic)
        - ami_sdm: AMI meeting corpus (single distant mic)
        - coraal: African American Language corpus
        - chime6: Multi-microphone conversation corpus
        - wsj: Wall Street Journal read speech
        - callhome: Conversational telephone speech
        - switchboard: Topic-guided conversation corpus

    Returns:
        None: Results are logged to specified output destinations

    Raises:
        FileNotFoundError: If checkpoint file or required data not found
        ValueError: If eval_set is not supported or invalid parameters
        RuntimeError: If CUDA requested but not available

    Example:
        >>> # Basic evaluation
        >>> short_form_eval(
        ...     batch_size=32,
        ...     num_workers=4,
        ...     ckpt="models/whisper_large.pt",
        ...     eval_set="librispeech_clean",
        ...     log_dir="results/"
        ... )
        >>>
        >>> # Training-linked evaluation with WandB
        >>> short_form_eval(
        ...     batch_size=16,
        ...     num_workers=8,
        ...     ckpt="checkpoints/step_5000.pt",
        ...     eval_set="tedlium",
        ...     log_dir="eval_logs/",
        ...     current_step=5000,
        ...     train_exp_name="whisper_training",
        ...     train_run_id="abc123",
        ...     wandb_log=True,
        ...     wandb_log_dir="wandb_logs/",
        ...     bootstrap=True
        ... )

    Output Files:
        - {log_dir}/eval_results.txt: Summary results (standalone evaluation)
        - {log_dir}/{train_exp_name}_{train_run_id}.txt: Training-linked results
        - {log_dir}/{eval_set}_sample_wer.csv: Per-sample WER (if bootstrap=True)

    Metrics Computed:
        - Word Error Rate (WER): Primary ASR evaluation metric
        - Substitutions: Number of word substitution errors
        - Insertions: Number of word insertion errors
        - Deletions: Number of word deletion errors

    Note:
        Checkpoints are automatically converted to inference format if needed.
        For training-linked evaluation, temporary inference checkpoints are
        cleaned up automatically after evaluation.
    """
    # Prepare checkpoint
    if "inf" not in ckpt and ckpt.split("/")[-2] != "whisper_ckpts":
        ckpt = gen_inf_ckpt(ckpt, ckpt.replace(".pt", "_inf.pt"))
    
    # get HF token
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")

    # Create directories
    for directory in [log_dir, wandb_log_dir, eval_dir]:
        if directory:
            os.makedirs(directory, exist_ok=True)

    device = torch.device("cuda" if cuda else "cpu")

    # Initialize dataset and dataloader
    dataset = EvalDataset(
        task="eng_transcribe",
        eval_set=eval_set,
        hf_token=hf_token,
        eval_dir=eval_dir,
        n_mels=n_mels,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    # Load model
    model = load_model(name=ckpt, device=device, inference=True, in_memory=True)
    model.eval()

    normalizer = EnglishTextNormalizer()
    hypotheses, references = [], []
    per_sample_wer = [] if bootstrap else None

    # Initialize wandb if needed
    eval_table = None
    if wandb_log:
        eval_table = WandBLogger.init_wandb(
            ckpt, eval_set, train_exp_name, train_run_id, run_id_dir, wandb_log_dir
        )

    # Main evaluation loop
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            audio_fp, audio_arr, audio_input, text_y = batch

            # Normalize and filter texts
            norm_tgt_text = [normalizer(text) for text in text_y]
            valid_indices = [
                i
                for i, text in enumerate(norm_tgt_text)
                if text and text != "ignore time segment in scoring"
            ]

            if not valid_indices:
                continue

            # Run inference
            audio_input = audio_input.to(device)
            options = DecodingOptions(language="en", without_timestamps=True)
            results = model.decode(audio_input, options=options)

            # Process results
            norm_pred_text = [normalizer(results[i].text) for i in valid_indices]
            norm_tgt_text = [norm_tgt_text[i] for i in valid_indices]

            references.extend(norm_tgt_text)
            hypotheses.extend(norm_pred_text)

            # Handle logging and bootstrap sampling
            if wandb_log and eval_table and (batch_idx + 1) // 10 == 1:
                audio_arr_filtered = [audio_arr.numpy()[i] for i in valid_indices]
                _add_to_wandb_table(
                    eval_table,
                    eval_set,
                    audio_arr_filtered,
                    norm_pred_text,
                    norm_tgt_text,
                )

                log_key = f"eval_table_{current_step}" if train_run_id else "eval_table"
                wandb.log({log_key: eval_table})

            elif bootstrap and per_sample_wer is not None:
                per_sample_wer.extend(
                    [
                        [
                            jiwer.wer(
                                reference=norm_tgt_text[i], hypothesis=norm_pred_text[i]
                            ),
                            len(norm_tgt_text[i]),
                        ]
                        for i in range(len(norm_pred_text))
                    ]
                )

    # Calculate final metrics
    avg_wer = jiwer.wer(references, hypotheses) * 100
    avg_measures = jiwer.compute_measures(truth=references, hypothesis=hypotheses)

    # Log results
    _log_results(
        eval_set,
        avg_wer,
        avg_measures,
        log_dir,
        wandb_log,
        train_run_id,
        train_exp_name,
        current_step,
        bootstrap,
        per_sample_wer,
    )

    # Cleanup
    if train_run_id is not None:
        os.remove(ckpt)


def long_form_eval(
    batch_size: int,
    num_workers: int,
    ckpt: str,
    eval_set: Literal[
        "tedlium",
        "meanwhile",
        "rev16",
        "earnings21",
        "earnings22",
        "coraal",
        "kincaid46",
    ],
    log_dir: str,
    n_mels: int = DEFAULT_N_MELS,
    bootstrap: bool = False,
    wandb_log: bool = False,
    wandb_log_dir: str = "wandb",
    eval_dir: str = "data/eval",
    hf_token: Optional[str] = None,
) -> None:
    """Evaluate ASR model performance on long-form transcription tasks.

    This evaluation function handles variable-length audio segments (typically longer
    than 30 seconds) that require different processing than short-form evaluation.
    It uses the model's built-in transcribe() method with beam search and timestamp
    generation for optimal long-form performance.

    Key differences from short-form evaluation:
    - No mel-spectrogram preprocessing (handled internally by transcribe())
    - Uses beam search decoding with best-of selection
    - Processes full audio length without padding/trimming
    - Enables timestamp generation for alignment verification
    - Single-sample batch processing for stability

    Args:
        batch_size (int): Number of audio samples per batch (typically 1 for long-form)
        num_workers (int): Number of worker processes for data loading
        ckpt (str): Path to model checkpoint file
        eval_set (Literal): Name of evaluation dataset (see supported datasets below)
        log_dir (str): Directory for saving evaluation results and logs
        n_mels (int, optional): Number of mel-spectrogram bins (unused in long-form).
            Defaults to DEFAULT_N_MELS.
        bootstrap (bool, optional): Enable bootstrap sampling for confidence intervals.
            Defaults to False.
        wandb_log (bool, optional): Enable Weights & Biases logging. Defaults to False.
        wandb_log_dir (str, optional): Directory for WandB logs. Defaults to "wandb".
        eval_dir (str, optional): Root directory for evaluation datasets.
            Defaults to "data/eval".
        hf_token (Optional[str], optional): HuggingFace token for private datasets.
            Defaults to None.

    Supported Datasets:
        - tedlium: TED talks with longer audio segments
        - meanwhile: Meanwhile podcast corpus for long-form evaluation
        - rev16: Rev.com transcription corpus subset
        - earnings21: Corporate earnings call transcripts (2021)
        - earnings22: Corporate earnings call transcripts (2022)
        - coraal: CORAAL long-form variant with extended segments
        - kincaid46: Extended version of Kincaid corpus

    Returns:
        None: Results are logged to specified output destinations

    Raises:
        FileNotFoundError: If checkpoint file or required data not found
        ValueError: If eval_set is not supported for long-form evaluation
        RuntimeError: If CUDA not available (required for long-form)

    Example:
        >>> # Basic long-form evaluation
        >>> long_form_eval(
        ...     batch_size=1,
        ...     num_workers=2,
        ...     ckpt="models/whisper_large.pt",
        ...     eval_set="tedlium",
        ...     log_dir="long_form_results/"
        ... )
        >>>
        >>> # With WandB logging and bootstrap sampling
        >>> long_form_eval(
        ...     batch_size=1,
        ...     num_workers=4,
        ...     ckpt="checkpoints/whisper_model.pt",
        ...     eval_set="earnings21",
        ...     log_dir="eval_logs/",
        ...     bootstrap=True,
        ...     wandb_log=True,
        ...     wandb_log_dir="wandb_logs/",
        ...     hf_token="hf_token_here"
        ... )

    Output Files:
        - {log_dir}/eval_results.txt: Summary evaluation results
        - {log_dir}/{eval_set}_sample_wer.csv: Per-sample WER (if bootstrap=True)

    Metrics Computed:
        - Word Error Rate (WER): Primary metric for long-form transcription
        - Substitutions: Number of word substitution errors
        - Insertions: Number of word insertion errors
        - Deletions: Number of word deletion errors

    Model Configuration:
        The function uses optimized settings for long-form transcription:
        - task="transcribe": Pure transcription without translation
        - language="en": English language constraint
        - without_timestamps=False: Enable timestamp generation
        - beam_size=5: Beam search for better quality
        - best_of=5: Multiple candidate selection

    Note:
        Long-form evaluation requires significantly more memory and compute time
        compared to short-form evaluation. CUDA is mandatory for practical
        performance. Batch size is typically set to 1 for memory efficiency.
    """
    # Prepare checkpoint
    if "inf" not in ckpt and ckpt.split("/")[-2] != "whisper_ckpts":
        ckpt = gen_inf_ckpt(ckpt, ckpt.replace(".pt", "_inf.pt"))

    # get HF token
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
        
    # Create directories
    for directory in [log_dir, wandb_log_dir, eval_dir]:
        os.makedirs(directory, exist_ok=True)

    device = torch.device("cuda")

    # Initialize dataset and dataloader
    dataset = EvalDataset(
        task="long_form_transcribe",
        eval_set=eval_set,
        hf_token=hf_token,
        eval_dir=eval_dir,
        n_mels=n_mels,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    # Load model
    model = load_model(name=ckpt, device=device, inference=True, in_memory=True)
    model.eval()

    normalizer = EnglishTextNormalizer()
    hypotheses, references = [], []
    per_sample_wer = [] if bootstrap else None

    # Initialize wandb if needed
    eval_table = None
    if wandb_log:
        eval_table = WandBLogger.init_wandb(ckpt, eval_set, wandb_log_dir=wandb_log_dir)

    # Main evaluation loop
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            _, audio_arr, _, text_y = batch

            norm_tgt_text = [normalizer(text) for text in text_y]
            valid_texts = [text for text in norm_tgt_text if text]

            if not valid_texts:
                continue

            # Run inference
            audio_arr = audio_arr.to(device)
            options = dict(
                task="transcribe",
                language="en",
                without_timestamps=False,
                beam_size=5,
                best_of=5,
            )
            results = model.transcribe(audio_arr[0], verbose=False, **options)

            norm_pred_text = [normalizer(results["text"]) for _ in valid_texts]

            print(f"{norm_pred_text=}")
            print(f"{valid_texts=}")

            references.extend(valid_texts)
            hypotheses.extend(norm_pred_text)

            # Handle logging and bootstrap sampling
            if wandb_log and eval_table and not bootstrap:
                audio_arr_cpu = [
                    audio_arr.cpu().numpy()[i] for i in range(len(valid_texts))
                ]
                _add_to_wandb_table(
                    eval_table, eval_set, audio_arr_cpu, norm_pred_text, valid_texts
                )
            elif bootstrap and per_sample_wer is not None:
                per_sample_wer.extend(
                    [
                        [
                            jiwer.wer(
                                reference=valid_texts[i], hypothesis=norm_pred_text[i]
                            ),
                            len(valid_texts[i]),
                        ]
                        for i in range(len(norm_pred_text))
                    ]
                )
            else:
                wer = (
                    np.round(
                        jiwer.wer(reference=valid_texts, hypothesis=norm_pred_text), 2
                    )
                    * 100
                )
                print(f"{wer=}")

        if wandb_log and eval_table:
            wandb.log({"eval_table": eval_table})

    # Calculate final metrics
    avg_wer = jiwer.wer(references, hypotheses) * 100
    avg_measures = jiwer.compute_measures(truth=references, hypothesis=hypotheses)

    # Log results
    _log_results(
        eval_set,
        avg_wer,
        avg_measures,
        log_dir,
        wandb_log,
        bootstrap=bootstrap,
        per_sample_wer=per_sample_wer,
    )


def _add_to_wandb_table(eval_table, eval_set, audio_arr, norm_pred_text, norm_tgt_text):
    """Add evaluation results to WandB table for detailed logging.

    Computes per-sample metrics and adds audio examples with predictions and
    ground truth to the WandB table for interactive visualization and analysis.

    Args:
        eval_table (wandb.Table): WandB table to add results to
        eval_set (str): Name of the evaluation dataset
        audio_arr (list): List of audio arrays for each sample
        norm_pred_text (list): List of normalized predicted transcripts
        norm_tgt_text (list): List of normalized ground truth transcripts

    Note:
        This function modifies the eval_table in place by adding new rows.
        Each row contains the dataset name, audio sample, predictions, targets,
        and detailed error metrics for analysis.
    """
    for i, pred_text in enumerate(norm_pred_text):
        wer = (
            np.round(jiwer.wer(reference=norm_tgt_text[i], hypothesis=pred_text), 2)
            * 100
        )
        measures = jiwer.compute_measures(truth=norm_tgt_text[i], hypothesis=pred_text)

        eval_table.add_data(
            eval_set,
            wandb.Audio(audio_arr[i], sample_rate=SAMPLE_RATE),
            pred_text,
            norm_tgt_text[i],
            measures["substitutions"],
            measures["deletions"],
            measures["insertions"],
            wer,
        )


def _log_results(
    eval_set: str,
    avg_wer: float,
    avg_measures: Dict[str, int],
    log_dir: str,
    wandb_log: bool,
    train_run_id: Optional[str] = None,
    train_exp_name: Optional[str] = None,
    current_step: Optional[int] = None,
    bootstrap: bool = False,
    per_sample_wer: Optional[list] = None,
):
    """Log evaluation results to multiple output destinations.

    Handles comprehensive logging of evaluation metrics to console, files, and
    WandB based on the evaluation configuration. Supports both standalone and
    training-integrated evaluation logging.

    Args:
        eval_set (str): Name of the evaluation dataset
        avg_wer (float): Average Word Error Rate as percentage
        avg_measures (Dict[str, int]): Dictionary containing detailed error counts
            with keys: "substitutions", "insertions", "deletions"
        log_dir (str): Directory for saving log files
        wandb_log (bool): Whether to log results to WandB
        train_run_id (Optional[str], optional): Training run ID for linked evaluation.
            Defaults to None.
        train_exp_name (Optional[str], optional): Training experiment name for
            file naming. Defaults to None.
        current_step (Optional[int], optional): Current training step for timestamping.
            Defaults to None.
        bootstrap (bool, optional): Whether bootstrap sampling was used. Defaults to False.
        per_sample_wer (Optional[list], optional): List of per-sample WER values and
            reference lengths for bootstrap analysis. Defaults to None.

    Output Destinations:
        1. Console: Always prints summary metrics
        2. Files: Writes to appropriate log files based on evaluation type
        3. WandB: Logs metrics and summaries if wandb_log=True
        4. Bootstrap CSV: Saves per-sample metrics if bootstrap=True

    File Outputs:
        - Standalone: {log_dir}/eval_results.txt
        - Training-linked: {log_dir}/{train_exp_name}_{train_run_id}.txt
        - Bootstrap: {log_dir}/{eval_set}_sample_wer.csv

    WandB Outputs:
        - Training-linked: Timestamped metrics with global_step
        - Standalone: Summary metrics in run summary

    Example:
        >>> measures = {"substitutions": 45, "insertions": 12, "deletions": 8}
        >>> _log_results(
        ...     eval_set="librispeech_clean",
        ...     avg_wer=5.2,
        ...     avg_measures=measures,
        ...     log_dir="results/",
        ...     wandb_log=True,
        ...     bootstrap=True,
        ...     per_sample_wer=[(0.05, 20), (0.03, 15), ...]
        ... )
    """
    avg_subs = avg_measures["substitutions"]
    avg_ins = avg_measures["insertions"]
    avg_dels = avg_measures["deletions"]

    print(
        f"{eval_set} WER: {avg_wer}, Average Subs: {avg_subs}, Average Ins: {avg_ins}, Average Dels: {avg_dels}"
    )

    # Save bootstrap results if needed
    if bootstrap and per_sample_wer:
        with open(f"{log_dir}/{eval_set}_sample_wer.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["wer", "ref_length"])
            writer.writerows(per_sample_wer)

    # Log to wandb or files
    if wandb_log and train_run_id is not None:
        wandb.log(
            {
                f"eval/{eval_set}_wer": avg_wer,
                f"eval/{eval_set}_subs": avg_subs,
                f"eval/{eval_set}_ins": avg_ins,
                f"eval/{eval_set}_dels": avg_dels,
                "global_step": current_step,
            }
        )
    elif not wandb_log and train_run_id is not None:
        with open(f"{log_dir}/{train_exp_name}_{train_run_id}.txt", "a") as f:
            f.write(
                f"Current step {current_step}, {eval_set} WER: {avg_wer}, Subs: {avg_subs}, Ins: {avg_ins}, Dels: {avg_dels}\n"
            )
    elif wandb_log and train_run_id is None:
        wandb.run.summary.update(
            {
                "avg_wer": avg_wer,
                "avg_subs": avg_subs,
                "avg_ins": avg_ins,
                "avg_dels": avg_dels,
            }
        )
    elif not wandb_log and train_run_id is None:
        with open(f"{log_dir}/eval_results.txt", "a") as f:
            f.write(
                f"{eval_set} WER: {avg_wer}, Subs: {avg_subs}, Ins: {avg_ins}, Dels: {avg_dels}\n"
            )


if __name__ == "__main__":
    Fire(
        {
            "short_form_eval": short_form_eval,
            "long_form_eval": long_form_eval,
        }
    )
