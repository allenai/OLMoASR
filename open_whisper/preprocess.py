# break audio files into 30-second segments paired with subset of transcript that occurs within that time segment
# audio is resampled to 16000 hz, 80-channel log-magnitude mel spectrogram, computed with a 25ms window and 10ms hop size
# spectrogram is a visual representation of spectrum of frequencies of a signal as it varies with time
# mel spectrogram is a spectrogram that is generated using the Mel scale, a perceptual scale of pitches judged by listeners to be equal in distance from one another
# 80-channel mel spectrogram is a mel spectrogram represented by 80 mel bins (frequency channels)
# feature normalization: globally scale input to be between [-1, 1] with approximate mean 0 across pre-training dataset
import os
import shutil
import subprocess
import pandas as pd
import multiprocessing
from tqdm import tqdm
from itertools import repeat
from typing import Union
from open_whisper import utils
import numpy as np


def download_transcript(
    video_id: str, lang_code: str, output_dir: str, sub_format: str = "srt"
) -> None:
    # to not redownload
    if os.path.exists(f"{output_dir}/{video_id}/{video_id}.{lang_code}.{sub_format}"):
        return None

    if lang_code == "unknown":
        lang_code = "en"

    command = [
        "yt-dlp",
        "--write-subs",
        "--no-write-auto-subs",
        "--skip-download",
        "--sub-format",
        f"{sub_format}",
        "--sub-langs",
        f"{lang_code},-live_chat",
        f"https://www.youtube.com/watch?v={video_id}",
        "-o",
        f"{output_dir}/%(id)s/%(id)s.%(ext)s",
    ]

    if sub_format == "srt":
        command.extend(["--convert-subs", "srt"])

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.exists(
        f"{output_dir}/{video_id}/{video_id}.{lang_code}.{sub_format}"
    ):
        with open(f"logs/data/failed_download_t.txt", "a") as f:
            f.write(f"{video_id}\n")
        return None


def parallel_download_transcript(args) -> None:
    download_transcript(*args)


def download_audio(video_id: str, output_dir: str, ext: str = "m4a") -> None:
    # to not redownload
    if os.path.exists(f"{output_dir}/{video_id}/{video_id}.{ext}"):
        return None

    command = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "-f",
        f"bestaudio[ext={ext}]",
        "--audio-quality",
        "0",
        "-o",
        f"{output_dir}/%(id)s/%(id)s.%(ext)s",
    ]

    if ext == "wav":
        command.extend(["--extract-audio", "--audio-format", "wav"])

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # if after downloading, the file doesn't exist
    if not os.path.exists(f"{output_dir}/{video_id}/{video_id}.{ext}"):
        with open(f"logs/data/failed_download_a.txt", "a") as f:
            f.write(f"{video_id}\n")
        return None


def parallel_download_audio(args) -> None:
    download_audio(*args)


def clean_transcript(file_path) -> Union[None, bool]:
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    if content.strip() == "":
        return None

    # Replace &nbsp; with a space or an empty string
    modified_content = content.replace("&nbsp;", " ")
    modified_content = modified_content.replace("\h", "")
    modified_content = modified_content.replace("\h\h", "")

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(modified_content)

    return True


def chunk_audio_transcript(transcript_file: str, audio_file: str) -> None:
    try:
        # if transcript or audio files doesn't exist
        if not os.path.exists(transcript_file):
            with open(f"logs/data/unchunked_pairs.txt", "a") as f:
                f.write(f"{transcript_file}\n")
            return None
        if not os.path.exists(audio_file):
            with open(f"logs/data/unchunked_pairs.txt", "a") as f:
                f.write(f"{audio_file}\n")
            return None

        transcript_ext = transcript_file.split(".")[-1]

        t_output_dir = "/".join(transcript_file.split("/")[:3]) + "/segments"
        a_output_dir = "/".join(audio_file.split("/")[:3]) + "/segments"
        os.makedirs(t_output_dir, exist_ok=True)
        os.makedirs(a_output_dir, exist_ok=True)

        cleaned_transcript = clean_transcript(transcript_file)
        if cleaned_transcript is None:
            with open(f"logs/data/empty_transcripts.txt", "a") as f:
                f.write(f"{transcript_file}\n")
            return None

        transcript, *_ = utils.TranscriptReader(transcript_file).read()

        # if transcript file is empty
        if transcript == {}:
            with open(f"logs/data/empty_transcript.txt", "a") as f:
                f.write(f"{transcript_file}\n")
            return None

        a = 0
        b = 0

        timestamps = list(transcript.keys())
        diff = 0
        init_diff = 0

        while a < len(transcript) + 1:
            init_diff = utils.calculate_difference(timestamps[a][0], timestamps[b][1])
            if init_diff < 30000:
                diff = init_diff
                b += 1
            else:
                # edge case (when transcript line is > 30s)
                if b == a:
                    with open(f"logs/data/faulty_transcripts.txt", "a") as f:
                        f.write(f"{t_output_dir.split('/')[-2]}\tindex: {b}\n")
                    return None

                # write transcript file
                utils.write_segment(
                    timestamps[a:b],
                    transcript,
                    t_output_dir,
                    transcript_ext,
                )

                utils.trim_audio(
                    audio_file,
                    timestamps[a][0],
                    timestamps[b - 1][1],
                    0,
                    0,
                    a_output_dir,
                )

                init_diff = 0
                diff = 0

                # checking for silence
                if timestamps[b][0] != timestamps[b - 1][1]:
                    silence_segments = (
                        utils.calculate_difference(
                            timestamps[b - 1][1], timestamps[b][0]
                        )
                        // 30000
                    )

                    for i in range(0, silence_segments + 1):
                        start = utils.adjust_timestamp(
                            timestamps[b - 1][1], (i * 30000)
                        )

                        if i == silence_segments:
                            end = timestamps[b][0]
                        else:
                            end = utils.adjust_timestamp(start, 30000)

                        utils.write_segment(
                            [
                                (
                                    start,
                                    end,
                                )
                            ],
                            None,
                            t_output_dir,
                            transcript_ext,
                        )
                        utils.trim_audio(
                            audio_file,
                            start,
                            end,
                            0,
                            0,
                            a_output_dir,
                        )

                a = b

            if b == len(transcript) and diff < 30000:
                # write transcript file
                utils.write_segment(
                    timestamps[a:b],
                    transcript,
                    t_output_dir,
                    transcript_ext,
                )

                utils.trim_audio(
                    audio_file,
                    timestamps[a][0],
                    timestamps[b - 1][1],
                    0,
                    0,
                    a_output_dir,
                )

                break

        with open(f"logs/data/chunked_pairs.txt", "a") as f:
            f.write(f"{audio_file.split('/')[-1].split('.')[0]}\n")

        os.remove(transcript_file)
        os.remove(audio_file)

    except Exception as e:
        with open(f"logs/data/failed_chunking.txt", "a") as f:
            f.write(f"{transcript_file}\t{audio_file}\t{e}\n")
        return None


def parallel_chunk_audio_transcript(args) -> None:
    chunk_audio_transcript(*args)
