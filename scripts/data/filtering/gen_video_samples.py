"""
Video Sample Generation Script

This script generates video samples from audio (numpy arrays) and transcript files.
It supports two modes:
1. Process all files in a data directory structure
2. Process a random sample from a samples dictionary file

The script creates videos with subtitles overlaid on audio, suitable for
previewing or demonstrating processed audio-transcript pairs.

Features:
- Parallel processing for multiple files
- Automatic subtitle generation from transcript files
- Support for both structured directory processing and random sampling
- Configurable output paths and formats
- Error handling and progress tracking

Usage:
    python gen_video_samples.py --wav_dir="/tmp/wav" --video_dir="/tmp/videos" --data_dir="/path/to/data"
    python gen_video_samples.py --wav_dir="/tmp/wav" --video_dir="/tmp/videos" --samples_dicts_file="/path/to/samples.jsonl"

Dependencies:
    - moviepy: Video processing and subtitle generation
    - scipy: Audio file writing
    - numpy: Audio data manipulation
    - tqdm: Progress tracking
    - fire: Command-line interface
"""

import os
import glob
import json
import shutil
import multiprocessing
from itertools import chain, repeat
from typing import List, Optional, Dict, Tuple, Union, Any, Callable
import random

import numpy as np
from scipy.io.wavfile import write
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import TextClip, ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.tools.subtitles import SubtitlesClip
from IPython.display import Video
from fire import Fire
from tqdm import tqdm

# Type aliases for better readability
FilePath = str
VideoInfo = Tuple[
    str, str, str, str
]  # (npy_file, transcript_file, output_file, wav_file)
ProcessResult = Union[str, Tuple[str, str, Exception]]  # Success URL or error tuple
SampleDict = Dict[str, Any]


def npy_to_wav(
    npy_file: FilePath, wav_file: FilePath, sample_rate: int = 16000
) -> None:
    """
    Convert numpy audio array to WAV file.

    Args:
        npy_file: Path to input numpy file containing audio data
        wav_file: Path to output WAV file
        sample_rate: Sample rate for the output WAV file (default: 16000)

    Raises:
        FileNotFoundError: If the numpy file doesn't exist
        ValueError: If the numpy array format is invalid
    """
    try:
        # Load the audio data from the npy file
        audio_data = np.load(npy_file)

        # Write to a wav file
        write(wav_file, sample_rate, audio_data)
    except Exception as e:
        raise ValueError(f"Failed to convert {npy_file} to WAV: {e}")


def create_subtitle_generator() -> Callable[[str], TextClip]:
    """
    Create a text clip generator function for subtitles.

    Returns:
        Function that creates TextClip objects with consistent styling
    """

    def generator(text: str) -> TextClip:
        return TextClip(
            font="/stage/Helvetica.ttf",
            text=text,
            font_size=24,
            color="white",
            vertical_align="center",
            horizontal_align="center",
        )

    return generator


def generate_video(
    npy_file: FilePath,
    transcript_file: FilePath,
    output_file: FilePath,
    wav_file: FilePath,
    sample_rate: int = 16000,
) -> ProcessResult:
    """
    Generate a video with subtitles and audio from numpy array and transcript.

    Args:
        npy_file: Path to numpy file containing audio data
        transcript_file: Path to transcript file (SRT format)
        output_file: Path for output video file
        wav_file: Path for temporary WAV file
        sample_rate: Sample rate for audio conversion (default: 16000)

    Returns:
        URL string on success, or tuple of (npy_file, transcript_file, exception) on error
    """
    try:
        # Convert the npy file to a wav file
        npy_to_wav(npy_file, wav_file, sample_rate)

        # Load the converted wav file as the audio
        audio = AudioFileClip(wav_file)

        # Check if transcript file is empty
        with open(transcript_file, "r") as f:
            transcript_content = f.read().strip()

        if not transcript_content:
            print("Silent segment")
            # Create a silent video with black background
            video = ColorClip(
                size=(1000, 1000), color=(0, 0, 0), duration=audio.duration
            )
            video.audio = audio  # Use direct assignment as in original code
            video.write_videofile(output_file, fps=24)
        else:
            # Create subtitle generator
            generator = create_subtitle_generator()

            # Create subtitles from the transcript file
            subtitles = SubtitlesClip(transcript_file, make_textclip=generator)
            subtitles = subtitles.with_position(
                ("center", "bottom")
            )  # Use with_position as in original

            # Create the video with subtitles and audio
            video = CompositeVideoClip(clips=[subtitles], size=(800, 420))
            video.audio = audio  # Use direct assignment as in original code
            video.write_videofile(output_file, fps=24, codec="libx264")

        # Clean up temporary wav file
        if os.path.exists(wav_file):
            os.remove(wav_file)

        return "http://localhost:8080" + output_file
    except Exception as e:
        return (npy_file, transcript_file, e)


def parallel_generate_video(args: VideoInfo) -> ProcessResult:
    """
    Wrapper function for parallel video generation.

    Args:
        args: Tuple of (npy_file, transcript_file, output_file, wav_file)

    Returns:
        Result from generate_video function
    """
    return generate_video(*args)


def view_video(output_file: FilePath) -> Video:
    """
    Create a Video object for display in Jupyter notebooks.

    Args:
        output_file: Path to the video file

    Returns:
        IPython Video object configured for display
    """
    return Video(output_file, width=800, height=420)


def gen_file_list(
    seg_dir: FilePath, video_dir: FilePath, wav_dir: FilePath
) -> List[VideoInfo]:
    """
    Generate list of file paths for video processing from a segment directory.

    Args:
        seg_dir: Directory containing .npy and .srt files
        video_dir: Output directory for video files
        wav_dir: Directory for temporary WAV files

    Returns:
        List of tuples containing (npy_file, srt_file, video_path, wav_file)

    Raises:
        ValueError: If npy and srt file counts don't match
    """
    video_seg_list = []
    npy_files = sorted(glob.glob(f"{seg_dir}/*.npy"))
    srt_files = sorted(glob.glob(f"{seg_dir}/*.srt"))

    if len(npy_files) != len(srt_files):
        raise ValueError(
            f"Mismatch in file counts: {len(npy_files)} npy files, {len(srt_files)} srt files"
        )

    for i in range(len(npy_files)):
        # Create video output path
        base_name = npy_files[i].split("/")[-1].split(".")[0].replace(":", ",")
        video_seg_path = os.path.join(
            video_dir, seg_dir.split("/")[-1], f"{base_name}.mp4"
        )
        os.makedirs(os.path.dirname(video_seg_path), exist_ok=True)

        # Create wav file path
        wav_file = f"{wav_dir}/{npy_files[i].split('.')[0]}.wav"
        os.makedirs(os.path.dirname(wav_file), exist_ok=True)

        video_seg_list.append((npy_files[i], srt_files[i], video_seg_path, wav_file))

    return video_seg_list


def parallel_gen_file_list(
    args: Tuple[FilePath, FilePath, FilePath]
) -> List[VideoInfo]:
    """
    Wrapper function for parallel file list generation.

    Args:
        args: Tuple of (seg_dir, video_dir, wav_dir)

    Returns:
        List of video processing information tuples
    """
    return gen_file_list(*args)


def open_dicts_file(samples_dicts_file: FilePath) -> List[SampleDict]:
    """
    Read sample dictionaries from a JSONL file.

    Args:
        samples_dicts_file: Path to JSONL file containing sample dictionaries

    Returns:
        List of sample dictionaries extracted from the file

    Raises:
        FileNotFoundError: If the samples file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(samples_dicts_file, "r") as f:
            samples_dicts = list(
                chain.from_iterable(
                    json_line.get("sample_dicts", [])
                    for json_line in map(json.loads, f)
                    if json_line.get("sample_dicts") is not None
                )
            )
        return samples_dicts
    except Exception as e:
        raise ValueError(f"Failed to read samples file {samples_dicts_file}: {e}")


def process_directory_data(
    data_dir: FilePath, video_dir: FilePath, wav_dir: FilePath
) -> List[VideoInfo]:
    """
    Process all segment directories in the data directory.

    Args:
        data_dir: Root directory containing segment subdirectories
        video_dir: Output directory for video files
        wav_dir: Directory for temporary WAV files

    Returns:
        List of video processing information tuples
    """
    seg_dirs = glob.glob(f"{data_dir}/*")

    with multiprocessing.Pool() as pool:
        video_seg_list = list(
            chain(
                *tqdm(
                    pool.imap_unordered(
                        parallel_gen_file_list,
                        zip(seg_dirs, repeat(video_dir), repeat(wav_dir)),
                    ),
                    total=len(seg_dirs),
                    desc="Processing directories",
                )
            )
        )

    return video_seg_list


def process_sample_data(
    samples_dicts_file: FilePath,
    video_dir: FilePath,
    wav_dir: FilePath,
    num_samples: int = 1000,
) -> List[VideoInfo]:
    """
    Process a random sample of data from samples dictionary file.

    Args:
        samples_dicts_file: Path to JSONL file containing sample dictionaries
        video_dir: Output directory for video files
        wav_dir: Directory for temporary WAV files
        num_samples: Number of samples to process (default: 1000)

    Returns:
        List of video processing information tuples
    """
    samples_dicts = open_dicts_file(samples_dicts_file)

    # Fix: Use random.sample instead of numpy.random.choice for list of dicts
    actual_samples = min(num_samples, len(samples_dicts))
    sample_segs = random.sample(samples_dicts, actual_samples)
    print(f"Selected {len(sample_segs)} samples from {len(samples_dicts)} total")
    print("First 5 samples:", sample_segs[:5])

    video_seg_list = []
    for d in sample_segs:
        original_path = d["audio"]
        # Fix: Handle case where video_dir might be None
        video_subdir = video_dir.split("/")[-1] if video_dir else "videos"
        output_path = (
            original_path.replace("ow_seg", video_subdir)
            .replace("npy", "mp4")
            .replace(":", ",")
        )

        wav_file = f"{wav_dir}/{d['audio'].split('.')[0]}.wav"
        video_seg_list.append((d["audio"], d["transcript"], output_path, wav_file))

        # Create necessary directories
        os.makedirs(os.path.dirname(wav_file), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    return video_seg_list


def process_videos(video_seg_list: List[VideoInfo]) -> List[str]:
    """
    Process all videos in parallel and return successful results.

    Args:
        video_seg_list: List of video processing information tuples

    Returns:
        List of successful result URLs
    """
    with multiprocessing.Pool() as pool:
        results = list(
            tqdm(
                pool.imap_unordered(parallel_generate_video, video_seg_list),
                total=len(video_seg_list),
                desc="Generating videos",
            )
        )

    # Separate successful results from errors
    errors = [r for r in results if isinstance(r, tuple)]
    successful_results = [r for r in results if isinstance(r, str)]

    if errors:
        print(f"Encountered {len(errors)} errors:")
        for error in errors:
            print(f"  Error processing {error[0]}: {error[2]}")

    return successful_results


def gen_video_samples(
    wav_dir: FilePath,
    video_dir: Optional[FilePath] = None,
    samples_dicts_file: Optional[FilePath] = None,
    data_dir: Optional[FilePath] = None,
    num_samples: int = 1000,
) -> List[str]:
    """
    Main function to generate video samples from audio and transcript files.

    Args:
        wav_dir: Directory for temporary WAV files
        video_dir: Output directory for video files (optional)
        samples_dicts_file: Path to JSONL file with sample dictionaries (optional)
        data_dir: Root directory containing segment subdirectories (optional)
        num_samples: Number of samples to process when using samples_dicts_file (default: 1000)

    Returns:
        List of URLs for successfully generated videos

    Raises:
        ValueError: If neither samples_dicts_file nor data_dir is provided
    """
    # Create necessary directories
    os.makedirs(wav_dir, exist_ok=True)
    if video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)

    # Determine processing mode and generate file list
    if data_dir is not None:
        if video_dir is None:
            raise ValueError("video_dir must be provided when using data_dir")
        print(f"Processing all files in data directory: {data_dir}")
        video_seg_list = process_directory_data(data_dir, video_dir, wav_dir)
    elif samples_dicts_file is not None:
        if video_dir is None:
            raise ValueError("video_dir must be provided when using samples_dicts_file")
        print(f"Processing {num_samples} samples from: {samples_dicts_file}")
        video_seg_list = process_sample_data(
            samples_dicts_file, video_dir, wav_dir, num_samples
        )
    else:
        raise ValueError("Either samples_dicts_file or data_dir must be provided")

    print(f"Generated {len(video_seg_list)} video processing tasks")
    print("First 5 tasks:", video_seg_list[:5])

    # Process videos
    successful_results = process_videos(video_seg_list)

    # Clean up temporary directory
    if os.path.exists(wav_dir):
        shutil.rmtree(wav_dir)

    print(f"Successfully generated {len(successful_results)} videos")
    return successful_results


if __name__ == "__main__":
    Fire(gen_video_samples)
