"""
OLMoASR Evaluation Dataset Downloader

This module provides automated downloading and preprocessing of evaluation datasets
for automatic speech recognition (ASR) systems. It handles the complexity of different
dataset formats, download sources, and preprocessing requirements to create a
standardized evaluation pipeline.

Key Features:
    - Automated download from multiple sources (OpenSLR, HuggingFace, direct URLs)
    - Dataset-specific preprocessing and format standardization
    - Parallel processing for large datasets (CHiME-6)
    - Automatic directory structure creation and file organization
    - Support for both academic and commercial evaluation corpora

Supported Datasets:
    - LibriSpeech (clean/other): Large-scale audiobook corpus
    - TEDLIUM: TED talks for long-form evaluation
    - Common Voice: Mozilla's crowdsourced speech corpus
    - Artie Bias Corpus: Demographic bias evaluation dataset
    - CORAAL: African American Language corpus (placeholder)
    - CHiME-6: Multi-microphone conversation challenge
    - AMI (IHM/SDM): Meeting corpus with multiple microphone setups
    - VoxPopuli: Parliamentary speech corpus
    - FLEURS: Multilingual few-shot learning corpus
    - Meanwhile: Podcast corpus for long-form evaluation
    - Rev16: Professional transcription corpus
    - Earnings21/22: Corporate earnings call datasets

Usage:
    # Command line interface
    python get_eval_set.py --eval_set librispeech_clean --eval_dir data/eval/
    
    # Python API
    from scripts.eval.get_eval_set import get_eval_set
    get_eval_set("tedlium", eval_dir="data/eval")
    
    # With HuggingFace token for private datasets
    get_eval_set("common_voice", eval_dir="data/eval", hf_token="your_token_here")

Requirements:
    - wget: For downloading files from web sources
    - tar: For extracting compressed archives
    - HuggingFace account: For some datasets (Common Voice, etc.)
    - Sufficient disk space: Some datasets are very large (Common Voice ~54GB)

Note:
    This script requires external tools (wget, tar) and internet connectivity.
    Some datasets require authentication tokens or manual agreement to terms.
"""

import subprocess
import os
from typing import Literal, Optional
from datasets import load_dataset
from fire import Fire
import shutil
import glob
import json
import multiprocessing
from tqdm import tqdm
from itertools import repeat
from pydub import AudioSegment
import tarfile
import pandas as pd
import numpy as np

# AMI Meeting Corpus evaluation session IDs
# These are the standardized meeting session identifiers used in the AMI corpus
# for evaluation. Each ID corresponds to a recorded meeting session with
# multiple speakers and microphone configurations.
AMI_IDS = [
    "EN2002a",  # English native speakers, meeting 2002, part a
    "EN2002b",  # English native speakers, meeting 2002, part b
    "EN2002c",  # English native speakers, meeting 2002, part c
    "EN2002d",  # English native speakers, meeting 2002, part d
    "ES2004a",  # Non-native English speakers, meeting 2004, part a
    "ES2004b",  # Non-native English speakers, meeting 2004, part b
    "ES2004c",  # Non-native English speakers, meeting 2004, part c
    "ES2004d",  # Non-native English speakers, meeting 2004, part d
    "IS1009a",  # Scenario meeting, 2009, part a
    "IS1009b",  # Scenario meeting, 2009, part b
    "IS1009c",  # Scenario meeting, 2009, part c
    "IS1009d",  # Scenario meeting, 2009, part d
    "TS3003a",  # Training session, 2003, part a
    "TS3003b",  # Training session, 2003, part b
    "TS3003c",  # Training session, 2003, part c
    "TS3003d",  # Training session, 2003, part d
]


def get_eval_set(
    eval_set: Literal[
        "librispeech_clean",
        "librispeech_other",
        "tedlium",
        "common_voice",
        "artie_bias_corpus",
        "coraal",
        "chime6",
        "ami_ihm",
        "ami_sdm",
        "voxpopuli",
        "fleurs",
        "meanwhile",
        "rev16",
        "earnings21",
        "earnings22",
    ],
    eval_dir: str = "data/eval",
    hf_token: Optional[str] = None,
) -> Optional[str]:
    """Download and prepare evaluation datasets for ASR evaluation.

    This function handles the automated downloading, extraction, and preprocessing
    of various speech recognition evaluation datasets. It creates standardized
    directory structures and handles dataset-specific format requirements.

    Args:
        eval_set (Literal): Name of the evaluation dataset to download. Must be one
            of the supported dataset names listed below.
        eval_dir (str, optional): Root directory where datasets will be stored.
            Creates subdirectories as needed. Defaults to "data/eval".
        hf_token (Optional[str], optional): HuggingFace authentication token required
            for private datasets like Common Voice. Get from https://huggingface.co/settings/tokens.
            Defaults to None.

    Returns:
        Optional[str]: Returns loaded HuggingFace dataset object for HF-hosted datasets,
            None for directly downloaded datasets.

    Supported Datasets:
        librispeech_clean: LibriSpeech test-clean subset (clean read speech)
            - Size: ~350MB compressed, ~2.5GB extracted
            - Content: High-quality audiobook recordings
            - Format: FLAC audio + text transcripts

        librispeech_other: LibriSpeech test-other subset (noisy read speech)
            - Size: ~300MB compressed, ~2.3GB extracted
            - Content: Lower-quality audiobook recordings
            - Format: FLAC audio + text transcripts

        tedlium: TEDLIUM Release 3 test set (TED talks)
            - Size: ~1.5GB compressed, ~8GB extracted
            - Content: TED conference presentations
            - Format: SPH audio + STM transcripts

        common_voice: Mozilla Common Voice v5.1 English test set
            - Size: ~54GB (full dataset download required)
            - Content: Crowdsourced speech recordings
            - Format: MP3 audio via HuggingFace
            - Requires: HuggingFace token

        artie_bias_corpus: Artie Bias Corpus for fairness evaluation
            - Size: ~100MB
            - Content: Demographically diverse speech samples
            - Format: Various audio formats + TSV metadata

        coraal: CORAAL African American Language corpus
            - Status: Not yet implemented
            - Content: Dialect-specific speech recordings

        chime6: CHiME-6 Challenge evaluation set
            - Size: ~15GB
            - Content: Multi-microphone conversation recordings
            - Format: WAV audio + JSON transcripts
            - Special: Automatic audio segmentation included

        ami_ihm: AMI Meeting Corpus (Individual Headset Microphone)
            - Size: ~2GB per session
            - Content: Meeting recordings with close-mic audio
            - Format: WAV audio + text transcripts

        ami_sdm: AMI Meeting Corpus (Single Distant Microphone)
            - Size: ~2GB per session
            - Content: Meeting recordings with distant microphone
            - Format: WAV audio + text transcripts

        voxpopuli: VoxPopuli English evaluation set
            - Size: Variable (HuggingFace dataset)
            - Content: European Parliament recordings
            - Format: Audio via HuggingFace

        fleurs: FLEURS English evaluation set
            - Size: Variable (HuggingFace dataset)
            - Content: Multilingual few-shot learning corpus
            - Format: Audio via HuggingFace

        meanwhile: Meanwhile podcast corpus
            - Size: Variable (HuggingFace dataset)
            - Content: Long-form podcast audio
            - Format: Audio via HuggingFace

        rev16: Rev.com transcription corpus
            - Size: Variable (HuggingFace dataset)
            - Content: Professional transcription samples
            - Format: Audio via HuggingFace

        earnings21: Corporate earnings calls 2021
            - Size: Variable (HuggingFace dataset)
            - Content: Business conference calls
            - Format: Audio via HuggingFace

        earnings22: Corporate earnings calls 2022
            - Size: Variable (HuggingFace dataset)
            - Content: Business conference calls
            - Format: Audio via HuggingFace

    Raises:
        NotImplementedError: If eval_set is "coraal" (not yet supported)
        subprocess.CalledProcessError: If download commands fail
        FileNotFoundError: If required external tools (wget, tar) are not available
        OSError: If insufficient disk space or permission issues
        ValueError: If eval_set is not in the supported list

    Examples:
        >>> # Download LibriSpeech clean test set
        >>> get_eval_set("librispeech_clean", eval_dir="data/eval")

        >>> # Download Common Voice with authentication
        >>> get_eval_set(
        ...     "common_voice",
        ...     eval_dir="data/eval",
        ...     hf_token="hf_your_token_here"
        ... )

        >>> # Download CHiME-6 with automatic preprocessing
        >>> get_eval_set("chime6", eval_dir="data/eval")

        >>> # Download AMI Individual Headset Microphone data
        >>> get_eval_set("ami_ihm", eval_dir="data/eval")

    Directory Structure Created:
        eval_dir/
        ├── librispeech_test_clean/          # LibriSpeech clean
        ├── librispeech_test_other/          # LibriSpeech other
        ├── artie-bias-corpus/               # Artie Bias Corpus
        ├── TEDLIUM_release-3/legacy/test/   # TEDLIUM
        ├── ami/ihm/ or ami/sdm/             # AMI corpus
        ├── chime6/                          # CHiME-6
        │   ├── audio/                       # Original recordings
        │   ├── segments/                    # Segmented audio clips
        │   └── transcripts/                 # JSON transcription files
        └── *HuggingFace cache dirs*         # For HF datasets

    Dependencies:
        External Tools:
            - wget: For downloading files from web sources
            - tar: For extracting compressed archives

        Python Packages:
            - datasets: HuggingFace datasets library
            - pydub: Audio processing for CHiME-6 segmentation
            - tqdm: Progress bars for long operations
            - pandas, numpy: Data processing utilities

    Note:
        - Some datasets require significant download time and disk space
        - CHiME-6 includes automatic audio segmentation which can be time-intensive
        - HuggingFace datasets are cached locally after first download
        - Common Voice requires accepting dataset terms on HuggingFace website
        - AMI corpus downloads multiple meeting sessions (16 sessions total)

    Performance Tips:
        - Use SSD storage for faster extraction and processing
        - Ensure stable internet connection for large downloads
        - Consider download resumption for interrupted transfers
        - Monitor disk space during CHiME-6 processing (temporary space needed)
    """
    os.makedirs(eval_dir, exist_ok=True)
    if eval_set == "librispeech_clean" and not os.path.exists(
        f"{eval_dir}/librispeech_test_clean"
    ):
        # downloading the file
        command = [
            "wget",
            "-P",
            eval_dir,
            "https://www.openslr.org/resources/12/test-clean.tar.gz",
        ]
        subprocess.run(command)
        # extracting the file
        command = ["tar", "-xvf", f"{eval_dir}/test-clean.tar.gz", "-C", eval_dir]
        subprocess.run(command)
        # removing the tar file
        os.remove(f"{eval_dir}/test-clean.tar.gz")
        # making test-clean main data folder
        os.rename(
            f"{eval_dir}/LibriSpeech/test-clean", f"{eval_dir}/librispeech_test_clean"
        )
        shutil.rmtree(f"{eval_dir}/LibriSpeech")
    elif eval_set == "librispeech_other":
        # downloading the file
        command = [
            "wget",
            "-P",
            eval_dir,
            "https://www.openslr.org/resources/12/test-other.tar.gz",
        ]
        subprocess.run(command)
        # extracting the file
        command = ["tar", "-xvf", f"{eval_dir}/test-other.tar.gz", "-C", eval_dir]
        subprocess.run(command)
        # removing the tar file
        os.remove(f"{eval_dir}/test-other.tar.gz")
        # making test-other main data folder
        os.rename(
            f"{eval_dir}/LibriSpeech/test-other", f"{eval_dir}/librispeech_test_other"
        )
        shutil.rmtree(f"{eval_dir}/LibriSpeech")
    elif eval_set == "artie_bias_corpus":
        # downloading the file
        command = [
            "wget",
            "-P",
            eval_dir,
            "http://ml-corpora.artie.com/artie-bias-corpus.tar.gz",
        ]
        subprocess.run(command)
        # extracting the file
        command = [
            "tar",
            "-xvf",
            f"{eval_dir}/artie-bias-corpus.tar.gz",
            "-C",
            eval_dir,
        ]
        subprocess.run(command)
        # removing the tar file
        os.remove(f"{eval_dir}/artie-bias-corpus.tar.gz")
    elif eval_set == "fleurs":
        dataset = load_dataset(
            path="google/fleurs",
            name="en_us",
            split="test",
            cache_dir=eval_dir,
            trust_remote_code=True,
            num_proc=15,
            save_infos=True,
        )
    # TODO: update + validate w/ HF's TEDLIUM dataset
    elif eval_set == "tedlium":
        # downloading the files
        command = [
            "wget",
            "-P",
            eval_dir,
            "https://huggingface.co/datasets/LIUM/tedlium/resolve/main/TEDLIUM_release3/legacy/test.tar.gz",
        ]
        subprocess.run(command)
        # extracting the file
        command = ["tar", "-xvf", f"{eval_dir}/test.tar.gz", "-C", eval_dir]
        subprocess.run(command)
        # removing the tar file
        os.remove(f"{eval_dir}/test.tar.gz")
        # renaming the folder
        os.makedirs(f"{eval_dir}/TEDLIUM_release-3/legacy", exist_ok=True)
        os.rename(f"{eval_dir}/test", f"{eval_dir}/TEDLIUM_release-3/legacy/test")
        os.makedirs(f"{eval_dir}/TEDLIUM_release-3/legacy/test/sph", exist_ok=True)
        os.makedirs(f"{eval_dir}/TEDLIUM_release-3/legacy/test/stm", exist_ok=True)
        for f in os.listdir(f"{eval_dir}/TEDLIUM_release-3/legacy/test"):
            if f.endswith(".stm"):
                os.rename(
                    f"{eval_dir}/TEDLIUM_release-3/legacy/test/{f}",
                    f"{eval_dir}/TEDLIUM_release-3/legacy/test/stm/{f}",
                )
            elif f.endswith(".sph"):
                os.rename(
                    f"{eval_dir}/TEDLIUM_release-3/legacy/test/{f}",
                    f"{eval_dir}/TEDLIUM_release-3/legacy/test/sph/{f}",
                )
    elif eval_set == "voxpopuli":
        dataset = load_dataset(
            path="facebook/voxpopuli",
            name="en",
            split="test",
            cache_dir=eval_dir,
            trust_remote_code=True,
            num_proc=15,
            save_infos=True,
        )
    elif eval_set == "common_voice":
        dataset = load_dataset(
            path="mozilla-foundation/common_voice_5_1",
            name="en",
            split="test",
            token=hf_token,
            cache_dir=eval_dir,
            trust_remote_code=True,
            num_proc=15,
            save_infos=True,
        )
    elif eval_set.startswith("ami"):
        ami_dir = f"{eval_dir}/ami"
        os.makedirs(ami_dir, exist_ok=True)
        if eval_set == "ami_ihm":
            ami_ihm_dir = f"{ami_dir}/ihm"
            os.makedirs(ami_ihm_dir, exist_ok=True)
            command = [
                "wget",
                "-P",
                ami_ihm_dir,
                "https://huggingface.co/datasets/edinburghcstr/ami/resolve/main/annotations/eval/text",
            ]
            subprocess.run(command)
            for _id in AMI_IDS:
                command = [
                    "wget",
                    "-P",
                    ami_ihm_dir,
                    f"https://huggingface.co/datasets/edinburghcstr/ami/resolve/main/audio/ihm/eval/{_id}.tar.gz",
                ]
                subprocess.run(command)
                command = [
                    "tar",
                    "-xvf",
                    f"{ami_ihm_dir}/{_id}.tar.gz",
                    "-C",
                    ami_ihm_dir,
                ]
                subprocess.run(command)
                os.remove(f"{ami_ihm_dir}/{_id}.tar.gz")
        elif eval_set == "ami_sdm":
            ami_sdm_dir = f"{ami_dir}/sdm"
            os.makedirs(ami_sdm_dir, exist_ok=True)
            command = [
                "wget",
                "-P",
                ami_sdm_dir,
                "https://huggingface.co/datasets/edinburghcstr/ami/resolve/main/annotations/eval/text",
            ]
            subprocess.run(command)
            for _id in AMI_IDS:
                command = [
                    "wget",
                    "-P",
                    ami_sdm_dir,
                    f"https://huggingface.co/datasets/edinburghcstr/ami/resolve/main/audio/sdm/eval/{_id}.tar.gz",
                ]
                subprocess.run(command)
                command = [
                    "tar",
                    "-xvf",
                    f"{ami_sdm_dir}/{_id}.tar.gz",
                    "-C",
                    ami_sdm_dir,
                ]
                subprocess.run(command)
                os.remove(f"{ami_sdm_dir}/{_id}.tar.gz")

            for root, dirs, files in os.walk(ami_sdm_dir):
                for f in files:
                    if "sdm" in f:
                        new_name = f.replace("sdm", "h00")
                        os.rename(f"{root}/{f}", f"{root}/{new_name}")
    elif eval_set == "chime6":
        eval_dir = os.path.join(eval_dir, "chime6")
        os.makedirs(eval_dir, exist_ok=True)
        command = [
            "wget",
            "-P",
            eval_dir,
            "https://www.openslr.org/resources/150/CHiME6_eval.tar.gz",
            "https://www.openslr.org/resources/150/CHiME6_transcriptions.tar.gz",
        ]
        subprocess.run(command)
        command = ["tar", "-xvf", f"{eval_dir}/CHiME6_eval.tar.gz", "-C", eval_dir]
        subprocess.run(command)
        command = [
            "tar",
            "-xvf",
            f"{eval_dir}/CHiME6_transcriptions.tar.gz",
            "-C",
            eval_dir,
        ]
        os.remove(f"{eval_dir}/CHiME6_eval.tar.gz")
        os.remove(f"{eval_dir}/CHiME6_transcriptions.tar.gz")

        os.rename(f"{eval_dir}/CHiME6_eval/CHiME6/audio/eval", f"{eval_dir}/audio")
        shutil.rmtree(f"{eval_dir}/CHiME6_eval")
        for p in glob.glob(f"{eval_dir}/audio/*_U*.wav"):
            os.remove(p)
        shutil.rmtree(f"{eval_dir}/transcriptions/transcriptions/dev")
        shutil.rmtree(f"{eval_dir}/transcriptions/transcriptions/train")
        os.rename(
            f"{eval_dir}/transcriptions/transcriptions/eval", f"{eval_dir}/transcripts"
        )
        shutil.rmtree(f"{eval_dir}/transcriptions")

        def timestamp_to_ms(timestamp):
            """Convert timestamp string to milliseconds for audio processing.

            Converts time format used in CHiME-6 transcripts (HH:MM:SS.mmm) to
            milliseconds for precise audio segmentation with pydub.

            Args:
                timestamp (str): Timestamp in "HH:MM:SS" or "HH:MM:SS.mmm" format
                    where HH=hours, MM=minutes, SS=seconds, mmm=milliseconds

            Returns:
                int: Timestamp converted to total milliseconds

            Example:
                >>> timestamp_to_ms("00:01:30.500")
                90500
                >>> timestamp_to_ms("00:00:05")
                5000
            """
            h, m, s = map(float, timestamp.split(":"))
            return int((h * 3600 + m * 60 + s) * 1000)

        def create_segment(src_dir, dst_dir, seg_dict):
            """Create audio segment from source audio based on timing information.

            Extracts a time-based segment from a source audio file and saves it
            as a separate WAV file. This is used to create individual utterance
            clips from longer CHiME-6 conversation recordings.

            Args:
                src_dir (str): Directory containing source audio files
                dst_dir (str): Directory where segmented clips will be saved
                seg_dict (dict): Dictionary containing segmentation metadata with keys:
                    - "audio_file": Source audio filename
                    - "audio_seg_file": Output segment filename
                    - "start_time": Segment start time (HH:MM:SS format)
                    - "end_time": Segment end time (HH:MM:SS format)

            Returns:
                str: Path to the created audio segment file

            Raises:
                FileNotFoundError: If source audio file doesn't exist
                OSError: If unable to create destination directory or write file
                ValueError: If timestamp format is invalid

            Example:
                >>> seg_dict = {
                ...     "audio_file": "S02_U06.wav",
                ...     "audio_seg_file": "S02_U06_0012500_0025000.wav",
                ...     "start_time": "00:00:12.5",
                ...     "end_time": "00:00:25.0"
                ... }
                >>> segment_path = create_segment("audio/", "segments/", seg_dict)

            Note:
                Creates destination directory if it doesn't exist. Uses pydub
                for audio processing, which supports various audio formats.
            """
            audio_file = os.path.join(src_dir, seg_dict["audio_file"])
            segment_file = os.path.join(dst_dir, seg_dict["audio_seg_file"])

            os.makedirs(dst_dir, exist_ok=True)
            audio = AudioSegment.from_wav(audio_file)
            start_time = timestamp_to_ms(seg_dict["start_time"])
            end_time = timestamp_to_ms(seg_dict["end_time"])
            clip = audio[start_time:end_time]
            clip.export(segment_file, format="wav")
            return segment_file

        def parallel_create_segment(args):
            """Wrapper function for parallel processing of audio segmentation.

            Unpacks arguments and calls create_segment for use with multiprocessing.
            This enables parallel processing of multiple audio segments to speed up
            the CHiME-6 preprocessing pipeline.

            Args:
                args (tuple): Tuple containing (src_dir, dst_dir, seg_dict) arguments
                    for create_segment function

            Returns:
                str: Path to the created audio segment file

            Example:
                >>> import multiprocessing
                >>> args_list = [(src_dir, dst_dir, seg_dict1), (src_dir, dst_dir, seg_dict2)]
                >>> with multiprocessing.Pool() as pool:
                ...     results = pool.map(parallel_create_segment, args_list)

            Note:
                This function is specifically designed for use with multiprocessing.Pool
                which requires functions that take a single argument.
            """
            return create_segment(*args)

        for p in glob.glob(f"{eval_dir}/transcripts/*.json"):
            with open(p, "r") as f:
                data = json.load(f)

            for d in data:
                start = timestamp_to_ms(d["start_time"])
                end = timestamp_to_ms(d["end_time"])
                d["audio_file"] = f"{d['session_id']}_{d['speaker']}.wav"
                d["audio_seg_file"] = (
                    f"{d['session_id']}_{d['speaker']}_{start:07}_{end:07}.wav"
                )

            with open(p, "w") as f:
                json.dump(data, f)

            with multiprocessing.Pool() as pool:
                res = list(
                    tqdm(
                        pool.imap_unordered(
                            parallel_create_segment,
                            zip(
                                repeat(f"{eval_dir}/audio"),
                                repeat(f"{eval_dir}/segments"),
                                data,
                            ),
                        ),
                        total=len(data),
                    )
                )
    elif eval_set == "coraal":
        raise NotImplementedError("CORAAL is not supported yet.")
    elif eval_set == "meanwhile":
        dataset = load_dataset(
            path="distil-whisper/meanwhile",
            split="test",
            cache_dir=eval_dir,
            trust_remote_code=True,
            num_proc=15,
            save_infos=True,
        )
    elif eval_set == "rev16":
        dataset = load_dataset(
            path="distil-whisper/rev16",
            name="whisper_subset",
            split="test",
            cache_dir=eval_dir,
            trust_remote_code=True,
            num_proc=15,
            save_infos=True,
        )
    elif eval_set == "earnings21":
        dataset = load_dataset(
            path="distil-whisper/earnings21",
            name="full",
            split="test",
            cache_dir=eval_dir,
            trust_remote_code=True,
            num_proc=15,
            save_infos=True,
        )
    elif eval_set == "earnings22":
        dataset = load_dataset(
            path="distil-whisper/earnings22",
            name="full",
            split="test",
            cache_dir=eval_dir,
            trust_remote_code=True,
            num_proc=15,
            save_infos=True,
        )
    return dataset


if __name__ == "__main__":
    Fire(get_eval_set)
