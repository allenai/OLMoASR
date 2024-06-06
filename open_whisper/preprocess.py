import os
import shutil
import subprocess
from typing import Literal, Optional, List, Tuple
import sys
from open_whisper import utils
from tempfile import TemporaryDirectory
import glob
import numpy as np

SEGMENT_COUNT_THRESHOLD = 60


def download_transcript(
    video_id: str,
    lang_code: str,
    output_dir: str,
    sub_format: Literal["srt", "vtt"] = "srt",
) -> Optional[None]:
    """Download transcript of a video from YouTube

    Download transcript of a video from YouTube using video ID and language code represented by video_id and lang_code respectively.
    If output_dir is provided, the transcript file will be saved in the specified directory.
    If sub_format is provided, the transcript file will be saved in the specified format.

    Args:
        video_id: YouTube video ID
        lang_code: Language code of the transcript
        output_dir: Directory to download the transcript file
        sub_format: Format of the subtitle file
    """
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

    result = subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )

    os.makedirs(f"metadata/{output_dir}", exist_ok=True)
    identifiers = [
        "unavailable",
        "private",
        "terminated",
        "removed",
        "country",
        "closed",
        "copyright",
        "members" "not available",
    ]
    if any(identifier in result.stderr for identifier in identifiers):
        with open(f"metadata/{output_dir}/unavailable_videos.txt", "a") as f:
            f.write(f"{video_id}\n")

    return None


def parallel_download_transcript(args) -> None:
    """Parallelized version of download_transcript function to work in multiprocessing context"""
    download_transcript(*args)


def download_audio(
    video_id: str, output_dir: str, ext: Literal["m4a", "wav"] = "m4a"
) -> Optional[str]:
    """Download audio of a video from YouTube

    Download audio of a video from YouTube using video ID and extension represented by video_id and ext respectively.
    If output_dir is provided, the audio file will be saved in the specified directory.
    If ext is provided, the audio file will be saved in the specified format.

    Args:
        video_id: YouTube video ID
        output_dir: Directory to download the audio file
        ext: Extension of the audio file
    """
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
        "--postprocessor-args",
        "ffmpeg:ffmpeg -ar 16000 -ac 1",
        "-o",
        f"{output_dir}/%(id)s/%(id)s.%(ext)s",
        "--downloader",
        "aria2c",
        "-N",
        "5",
    ]

    if ext == "wav":
        command.extend(["--extract-audio", "--audio-format", "wav"])

    result = subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )

    os.makedirs(f"metadata/{output_dir}", exist_ok=True)
    identifiers = [
        "unavailable",
        "private",
        "terminated",
        "removed",
        "country",
        "closed",
        "copyright",
        "members",
        "not available",
    ]
    if any(identifier in result.stderr for identifier in identifiers):
        with open(f"metadata/{output_dir}/unavailable_videos.txt", "a") as f:
            f.write(f"{video_id}\n")
            return None
    elif "HTTP Error 403" in result.stderr:
        with open(f"metadata/{output_dir}/blocked_ip.txt", "a") as f:
            f.write(f"{video_id}\n")
        return "requeue"

    return None


def parallel_download_audio(args) -> None:
    """Parallelized version of download_audio function to work in multiprocessing context"""
    download_audio(*args)


def chunk_audio_transcript(
    transcript_file: str, audio_file: str, output_dir: str, in_memory: bool
) -> Optional[List[Tuple[str, str, str, np.ndarray]]]:
    """Segment audio and transcript files into <= 30-second chunks

    Segment audio and transcript files into <= 30-second chunks. The audio and transcript files are represented by audio_file and transcript_file respectively.

    Args:
    transcript_file: Path to the transcript file
    audio_file: Path to the audio file

    Raises:
        Exception: If an error occurs during the chunking process
    """
    video_id_dir = "/".join(transcript_file.split("/")[:-1])
    # already processed
    if not os.path.exists(video_id_dir):
        return None

    log_dir = "logs/data/preprocess"
    os.makedirs("logs/data/preprocess", exist_ok=True)
    faulty_dir = "data/untrainable/faulty_transcripts"
    os.makedirs(faulty_dir, exist_ok=True)
    failed_dir = "data/untrainable/failed_chunking"
    os.makedirs(failed_dir, exist_ok=True)
    uneven_dir = "data/untrainable/uneven_chunks"
    os.makedirs(uneven_dir, exist_ok=True)

    try:
        # debug
        with open(os.path.join(log_dir, "processing.txt"), "a") as f:
            f.write(f"{video_id_dir.split('/')[-1]}\n")
        print(f"Processing {video_id_dir.split('/')[-1]}")
        segment_output_dir = os.path.join(output_dir, transcript_file.split("/")[-2])
        if not in_memory:
            os.makedirs(segment_output_dir, exist_ok=True)
        transcript_ext = transcript_file.split(".")[-1]
        segment_count = 0

        # if transcript file is empty (1st ver)
        cleaned_transcript = utils.clean_transcript(transcript_file)
        if cleaned_transcript is None:
            with open(f"{log_dir}/empty_transcripts.txt", "a") as f:
                f.write(f"{video_id_dir.split('/')[-1]}\n")
            shutil.rmtree(video_id_dir)
            return None

        transcript, *_ = utils.TranscriptReader(transcript_file).read()

        # if transcript file is empty (2nd ver)
        if len(transcript.keys()) == 0:
            with open(f"{log_dir}/empty_transcripts.txt", "a") as f:
                f.write(f"{video_id_dir.split('/')[-1]}\n")
            shutil.rmtree(video_id_dir)
            return None

        a = 0
        b = 0

        timestamps = list(transcript.keys())
        diff = 0
        init_diff = 0
        segments_list = []

        while a < len(transcript) + 1 and segment_count < SEGMENT_COUNT_THRESHOLD:
            init_diff = utils.calculate_difference(timestamps[a][0], timestamps[b][1])

            if init_diff < 30000:
                diff = init_diff
                b += 1
            else:
                # edge case (when transcript line is > 30s)
                if b == a:
                    with open(
                        os.path.join(log_dir, "faulty_transcripts.txt"), "a"
                    ) as f:
                        f.write(f"{video_id_dir.split('/')[-1]}\tindex: {b}\n")

                    a += 1
                    b += 1

                    if a == b == len(transcript):
                        if segment_count == 0:
                            shutil.rmtree(video_id_dir)
                        break

                    continue

                if not utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript
                ):
                    t_output_file, transcript_string = utils.write_segment(
                        timestamps=timestamps[a:b],
                        transcript=transcript,
                        output_dir=segment_output_dir,
                        ext=transcript_ext,
                        in_memory=in_memory,
                    )

                    a_output_file, audio_arr = utils.trim_audio(
                        audio_file=audio_file,
                        start=timestamps[a][0],
                        end=timestamps[b - 1][1],
                        output_dir=segment_output_dir,
                        in_memory=in_memory,
                    )

                    if audio_arr is None:
                        continue
                    elif utils.too_short_audio(audio_arr=audio_arr):
                        continue
                    else:
                        segments_list.append(
                            (t_output_file, transcript_string, a_output_file, audio_arr)
                        )
                        segment_count += 1

                init_diff = 0
                diff = 0

                # checking for silence
                if timestamps[b][0] > timestamps[b - 1][1]:
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
                            if start == timestamps[b][0]:
                                continue
                            else:
                                end = timestamps[b][0]
                        else:
                            end = utils.adjust_timestamp(start, 30000)

                        t_output_file, transcript_string = utils.write_segment(
                            timestamps=[(start, end)],
                            transcript=None,
                            output_dir=segment_output_dir,
                            ext=transcript_ext,
                            in_memory=in_memory,
                        )

                        a_output_file, audio_arr = utils.trim_audio(
                            audio_file=audio_file,
                            start=start,
                            end=end,
                            output_dir=segment_output_dir,
                            in_memory=in_memory,
                        )

                        if audio_arr is None:
                            continue
                        elif utils.too_short_audio(audio_arr=audio_arr):
                            continue
                        else:
                            segments_list.append(
                                (
                                    t_output_file,
                                    transcript_string,
                                    a_output_file,
                                    audio_arr,
                                )
                            )
                            segment_count += 1

                a = b

            if b == len(transcript) and diff < 30000:
                if not utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript
                ):
                    t_output_file, transcript_string = utils.write_segment(
                        timestamps=timestamps[a:b],
                        transcript=transcript,
                        output_dir=segment_output_dir,
                        ext=transcript_ext,
                        in_memory=in_memory,
                    )

                    a_output_file, audio_arr = utils.trim_audio(
                        audio_file=audio_file,
                        start=timestamps[a][0],
                        end=timestamps[b - 1][1],
                        output_dir=segment_output_dir,
                        in_memory=in_memory,
                    )

                    if audio_arr is None:
                        break
                    elif utils.too_short_audio(audio_arr=audio_arr):
                        break
                    else:
                        segments_list.append(
                            (t_output_file, transcript_string, a_output_file, audio_arr)
                        )
                        segment_count += 1

                break
        print(f"Finished {video_id_dir.split('/')[-1]}")
        return segments_list
    except ValueError as e:
        with open(os.path.join(log_dir, "failed_chunking.txt"), "a") as f:
            f.write(f"{transcript_file}\t{audio_file}\t{e}\n")
        shutil.move(video_id_dir, failed_dir)
        return None
    except Exception as e:
        with open(os.path.join(log_dir, "failed_chunking.txt"), "a") as f:
            f.write(f"{transcript_file}\t{audio_file}\t{e}\n")
        shutil.move(video_id_dir, failed_dir)
        return None


def parallel_chunk_audio_transcript(args) -> None:
    """Parallelized version of chunk_audio_transcript function to work in multiprocessing context"""
    chunk_audio_transcript(*args)
