import os
import shutil
import subprocess
from typing import Literal, Optional, List, Tuple, Dict
import sys
from open_whisper import utils
from tempfile import TemporaryDirectory
import glob
import numpy as np
import tarfile
from io import BytesIO
from tqdm import tqdm

SEGMENT_COUNT_THRESHOLD = 120


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
        with open(f"metadata/unavailable_videos.txt", "a") as f:
            f.write(f"{video_id}\t{output_dir}\ttranscript\n")
        return "unavailable"

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
        with open(f"metadata/unavailable_videos.txt", "a") as f:
            f.write(f"{video_id}\t{output_dir}\taudio\n")
            return "unavailable"
    elif (
        "HTTP Error 403" in result.stderr
        or "Only images are available for download" in result.stderr
        or "Requested format is not available" in result.stderr
    ):
        with open(f"metadata/blocked_ip.txt", "a") as f:
            f.write(f"{video_id}\t{output_dir}\n")
        return "blocked IP"

    return None


def parallel_download_audio(args) -> None:
    """Parallelized version of download_audio function to work in multiprocessing context"""
    download_audio(*args)


def read_file_in_tar(
    tar_gz_path, file_in_tar: str, audio_dir: Optional[str]
) -> Optional[str]:
    """Reads and returns the content of a specific file within a .tar.gz archive."""
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        binary_content = tar.extractfile(file_in_tar).read()
        if file_in_tar.endswith("srt") or file_in_tar.endswith("vtt"):
            file_content = binary_content.decode("utf-8")
            return file_content
        elif file_in_tar.endswith("m4a"):
            output_path = f"{audio_dir}/{file_in_tar.split('/')[-1]}"
            with open(output_path, "wb") as f:
                f.write(binary_content)
            return output_path
        else:
            return None


def chunk_data(
    transcript: Dict,
    transcript_ext: str,
    audio_file: Optional[str] = None,
    segment_output_dir: Optional[str] = None,
    video_id: Optional[str] = None,
    language: Optional[str] = None,
    audio_only: bool = False,
    transcript_only: bool = False,
    dolma_format: bool = False,
    in_memory: bool = True,
    on_gcs: bool = False,
    log_dir: Optional[str] = None,
) -> Optional[Tuple[List, int, int, int, int, int]]:
    a = 0
    b = 0

    segment_count = 0
    over_30_line_segment_count = 0
    bad_text_segment_count = 0
    over_ctx_len_segment_count = 0
    faulty_audio_segment_count = 0
    faulty_transcript_count = 0
    failed_transcript_count = 0

    timestamps = list(transcript.keys())
    diff = 0
    init_diff = 0
    segments_list = []

    try:
        while a < len(transcript) + 1 and segment_count < SEGMENT_COUNT_THRESHOLD:
            init_diff = utils.calculate_difference(timestamps[a][0], timestamps[b][1])

            if init_diff <= 30000:
                diff = init_diff
                b += 1
            # init_diff > 30000
            else:
                # edge case (when transcript line is > 30s), not included in segment group
                if b == a:
                    over_30_line_segment_count += 1

                    if on_gcs:
                        with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")

                    a += 1
                    b += 1

                    # if reach end of transcript or trancscript only has 1 line
                    if a == b == len(transcript):
                        # if transcript has only 1 line that is > 30s, stop processing
                        if segment_count == 0:
                            faulty_transcript_count += 1
                        break

                    continue

                over_ctx_len, err = utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript, language=language
                )
                # check if segment text goes over model context length
                if not over_ctx_len:
                    if transcript_only is True:
                        # writing transcript segment w/ timestamps[a][0] -> timestamps[b - 1][1]
                        t_output_file, transcript_string, norm_end = utils.write_segment(
                            timestamps=timestamps[a:b],
                            transcript=transcript,
                            output_dir=segment_output_dir,
                            ext=transcript_ext,
                            in_memory=in_memory,
                        )

                        if not utils.too_short_audio_text(
                            start=timestamps[a][0], end=utils.adjust_timestamp(timestamps[a][0], 30000)
                        ):
                            audio_timestamp = f"{timestamps[a][0].replace('.', ',')}_{utils.adjust_timestamp(timestamps[a][0], 30000).replace('.', ',')}"
                            text_timestamp = t_output_file.split("/")[-1].split(
                                    f".{transcript_ext}"
                                )[0]
                            if dolma_format is False:
                                outputs = {
                                    "subtitle_file": t_output_file.replace(
                                        "ow_full", "ow_seg"
                                    ),
                                    "seg_content": transcript_string,
                                    "text_timestamp": text_timestamp,
                                    "audio_timestamp": audio_timestamp,
                                    "norm_text_end": norm_end,
                                    "id": video_id,
                                    "seg_id": f"{video_id}_{segment_count}",
                                    "audio_file": os.path.join(os.path.dirname(t_output_file), f"{audio_timestamp}.npy").replace("ow_full", "ow_seg"),
                                }
                            else:
                                outputs = {
                                    "id": f"{video_id}_{segment_count}",
                                    "text": transcript_string,
                                    "source": "OW",
                                    "metadata": {
                                        "subtitle_file": t_output_file.replace(
                                            "ow_full", "ow_seg"
                                        ),
                                        "text_timestamp": text_timestamp,
                                        "audio_timestamp": audio_timestamp,
                                        "norm_text_end": norm_end,
                                        "audio_file": os.path.join(os.path.dirname(t_output_file), f"{audio_timestamp}.npy").replace("ow_full", "ow_seg"),
                                    },
                                }

                            segments_list.append(outputs)
                            segment_count += 1
                    elif audio_only is True:
                        # writing audio segment w/ timestamps[a][0] -> timestamps[a][0] + 30s
                        a_output_file, audio_arr = utils.trim_audio(
                            audio_file=audio_file,
                            start=timestamps[a][0],
                            end=utils.adjust_timestamp(timestamps[a][0], 30000),
                            output_dir=segment_output_dir,
                            in_memory=in_memory,
                        )
                    elif transcript_only is False and audio_only is False:
                        # writing transcript segment w/ timestamps[a][0] -> timestamps[b - 1][1]
                        t_output_file, transcript_string, _ = utils.write_segment(
                            timestamps=timestamps[a:b],
                            transcript=transcript,
                            output_dir=segment_output_dir,
                            ext=transcript_ext,
                            in_memory=in_memory,
                        )

                        # writing audio segment w/ timestamps[a][0] -> timestamps[b - 1][1]
                        a_output_file, audio_arr = utils.trim_audio(
                            audio_file=audio_file,
                            start=timestamps[a][0],
                            end=utils.adjust_timestamp(timestamps[a][0], 30000),
                            output_dir=segment_output_dir,
                            in_memory=in_memory,
                        )

                    if audio_only is True or (
                        transcript_only is False and audio_only is False
                    ):
                        # check if audio segment is too short or that audio array is valid (not None)
                        if audio_arr is not None and not utils.too_short_audio(
                            audio_arr=audio_arr
                        ):
                            if audio_only:
                                outputs = (a_output_file, audio_arr)
                            else:
                                outputs = (
                                    t_output_file,
                                    transcript_string,
                                    a_output_file,
                                    audio_arr,
                                )
                            segments_list.append(outputs)
                            segment_count += 1
                        else:
                            if audio_arr is None:
                                if on_gcs:
                                    with open(f"{log_dir}/faulty_audio.txt", "a") as f:
                                        f.write(f"{video_id}\tindex: {b}\n")
                                else:
                                    faulty_audio_segment_count += 1
                else:
                    if err is not None:
                        if on_gcs:
                            with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                        bad_text_segment_count += 1
                    else:
                        if on_gcs:
                            with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                        over_ctx_len_segment_count += 1

                init_diff = 0
                diff = 0

                # checking for no-speech segments
                if timestamps[b][0] > timestamps[b - 1][1]:
                    no_speech_segments = (
                        utils.calculate_difference(
                            timestamps[b - 1][1], timestamps[b][0]
                        )
                        // 30000
                    )

                    for i in range(0, no_speech_segments + 1):
                        start = utils.adjust_timestamp(
                            timestamps[b - 1][1], (i * 30000)
                        )

                        if i == no_speech_segments:
                            if start == timestamps[b][0]:
                                continue
                            else:
                                end = timestamps[b][0]
                        else:
                            end = utils.adjust_timestamp(start, 30000)

                        if transcript_only is True:
                            t_output_file, transcript_string, _ = utils.write_segment(
                                timestamps=[(start, end)],
                                transcript=None,
                                output_dir=segment_output_dir,
                                ext=transcript_ext,
                                in_memory=in_memory,
                            )

                            if not utils.too_short_audio_text(start=start, end=end):
                                timestamp = t_output_file.split("/")[-1].split(
                                    f".{transcript_ext}"
                                )[0]

                                if dolma_format is False:
                                    outputs = {
                                        "subtitle_file": t_output_file.replace(
                                            "ow_full", "ow_seg"
                                        ),
                                        "seg_content": transcript_string,
                                        "text_timestamp": timestamp,
                                        "audio_timestamp": timestamp,
                                        "norm_text_end": None,
                                        "id": video_id,
                                        "seg_id": f"{video_id}_{segment_count}",
                                        "audio_file": t_output_file.replace(
                                            f".{transcript_ext}", ".npy"
                                        ).replace("ow_full", "ow_seg"),
                                    }
                                else:
                                    outputs = {
                                        "id": f"{video_id}_{segment_count}",
                                        "text": transcript_string,
                                        "source": "OW",
                                        "metadata": {
                                            "subtitle_file": t_output_file.replace(
                                                "ow_full", "ow_seg"
                                            ),
                                            "text_timestamp": timestamp,
                                            "audio_timestamp": timestamp,
                                            "norm_text_end": None,
                                            "audio_file": t_output_file.replace(
                                                f".{transcript_ext}", ".npy"
                                            ).replace("ow_full", "ow_seg"),
                                        },
                                    }
                                segments_list.append(outputs)
                                segment_count += 1
                        elif audio_only is True:
                            a_output_file, audio_arr = utils.trim_audio(
                                audio_file=audio_file,
                                start=start,
                                end=end,
                                output_dir=segment_output_dir,
                                in_memory=in_memory,
                            )
                        elif transcript_only is False and audio_only is False:
                            t_output_file, transcript_string, _ = utils.write_segment(
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

                        if audio_only is True or (
                            transcript_only is False and audio_only is False
                        ):
                            if audio_arr is not None and not utils.too_short_audio(
                                audio_arr=audio_arr
                            ):
                                if audio_only:
                                    outputs = (a_output_file, audio_arr)
                                else:
                                    outputs = (
                                        t_output_file,
                                        transcript_string,
                                        a_output_file,
                                        audio_arr,
                                    )

                                segments_list.append(outputs)
                                segment_count += 1
                            else:
                                if audio_arr is None:
                                    if on_gcs:
                                        with open(
                                            f"{log_dir}/faulty_audio.txt", "a"
                                        ) as f:
                                            f.write(f"{video_id}\tindex: {b}\n")
                                    else:
                                        faulty_audio_segment_count += 1
                # moving pointer a to b
                a = b

            # at the end of transcript
            if b == len(transcript) and diff <= 30000:
                over_ctx_len, err = utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript, language=language
                )
                if not over_ctx_len:
                    if transcript_only is True:
                        t_output_file, transcript_string, norm_end = utils.write_segment(
                            timestamps=timestamps[a:b],
                            transcript=transcript,
                            output_dir=segment_output_dir,
                            ext=transcript_ext,
                            in_memory=in_memory,
                        )

                        if not utils.too_short_audio_text(
                            start=timestamps[a][0], end=timestamps[b - 1][1]
                        ):
                            timestamp = t_output_file.split("/")[-1].split(
                                f".{transcript_ext}"
                            )[0]
                            if dolma_format is False:
                                outputs = {
                                    "subtitle_file": t_output_file.replace(
                                        "ow_full", "ow_seg"
                                    ),
                                    "seg_content": transcript_string,
                                    "text_timestamp": timestamp,
                                    "audio_timestamp": timestamp,
                                    "norm_text_end": norm_end,
                                    "id": video_id,
                                    "seg_id": f"{video_id}_{segment_count}",
                                    "audio_file": t_output_file.replace(
                                        f".{transcript_ext}", ".npy"
                                    ).replace("ow_full", "ow_seg"),
                                }
                            else:
                                outputs = {
                                    "id": f"{video_id}_{segment_count}",
                                    "text": transcript_string,
                                    "source": "OW",
                                    "metadata": {
                                        "subtitle_file": t_output_file.replace(
                                            "ow_full", "ow_seg"
                                        ),
                                        "text_timestamp": timestamp,
                                        "audio_timestamp": timestamp,
                                        "norm_text_end": norm_end,
                                        "audio_file": t_output_file.replace(
                                            f".{transcript_ext}", ".npy"
                                        ).replace("ow_full", "ow_seg"),
                                    },
                                }
                            segments_list.append(outputs)
                            segment_count += 1
                    elif audio_only is True:
                        a_output_file, audio_arr = utils.trim_audio(
                            audio_file=audio_file,
                            start=timestamps[a][0],
                            end=timestamps[b - 1][1],
                            output_dir=segment_output_dir,
                            in_memory=in_memory,
                        )
                    elif transcript_only is False and audio_only is False:
                        t_output_file, transcript_string, _ = utils.write_segment(
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

                    if audio_only is True or (
                        transcript_only is False and audio_only is False
                    ):
                        if audio_arr is not None and not utils.too_short_audio(
                            audio_arr=audio_arr
                        ):
                            if audio_only:
                                outputs = (a_output_file, audio_arr)
                            else:
                                outputs = (
                                    t_output_file,
                                    transcript_string,
                                    a_output_file,
                                    audio_arr,
                                )
                            segments_list.append(outputs)
                            segment_count += 1
                        else:
                            if audio_arr is None:
                                if on_gcs:
                                    with open(f"{log_dir}/faulty_audio.txt", "a") as f:
                                        f.write(f"{video_id}\tindex: {b}\n")
                                else:
                                    faulty_audio_segment_count += 1
                else:
                    if err is not None:
                        if on_gcs:
                            with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                        bad_text_segment_count += 1
                    else:
                        if on_gcs:
                            with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                        over_ctx_len_segment_count += 1

                break

        if len(segments_list) == 0:
            return (
                None,
                over_30_line_segment_count,
                bad_text_segment_count,
                over_ctx_len_segment_count,
                faulty_audio_segment_count,
                faulty_transcript_count,
                failed_transcript_count
            )

        return (
            segments_list,
            over_30_line_segment_count,
            bad_text_segment_count,
            over_ctx_len_segment_count,
            faulty_audio_segment_count,
            faulty_transcript_count,
            failed_transcript_count
        )
    except ValueError as e:
        failed_transcript_count += 1
        if on_gcs:
            with open(f"{log_dir}/failed_chunking.txt", "a") as f:
                f.write(f"{video_id}\t{e}\n")
        return None, 0, 0, 0, 0, 0, failed_transcript_count
    except Exception as e:
        failed_transcript_count += 1
        if on_gcs:
            with open(f"{log_dir}/failed_chunking.txt", "a") as f:
                f.write(f"{video_id}\t{e}\n")
        return None, 0, 0, 0, 0, 0, failed_transcript_count


def chunk_local(
    transcript_file: str,
    audio_file: str,
    output_dir: str,
    audio_only: bool,
    transcript_only: bool,
    in_memory: bool,
):
    language = None  # temporarily
    os.makedirs(output_dir, exist_ok=True)
    video_id = os.path.dirname(transcript_file).split("/")[-1]
    segment_output_dir = os.path.join(output_dir, video_id)
    os.makedirs(segment_output_dir, exist_ok=True)
    transcript_ext = transcript_file.split(".")[-1]

    transcript, *_ = utils.TranscriptReader(
        file_path=transcript_file, transcript_string=None, ext=transcript_ext
    ).read()

    empty_transcript = 0
    if len(transcript.keys()) == 0:
        empty_transcript += 1
        return None, empty_transcript

    result = chunk_data(
        transcript=transcript,
        transcript_ext=transcript_ext,
        audio_file=audio_file,
        segment_output_dir=segment_output_dir,
        language=language,
        audio_only=audio_only,
        transcript_only=transcript_only,
        in_memory=in_memory,
    )

    return result


def chunk_gcs(
    tar_gz_file: str,
    transcript_file: str,
    audio_file: str,
    log_dir: str,
    audio_only: bool = True,
    in_memory: bool = True,
):
    video_id = transcript_file.split("/")[1]

    if "_" in tar_gz_file:
        language = tar_gz_file.split("_")[-1].split(".")[0]
    else:
        language = None

    transcript_string = read_file_in_tar(tar_gz_file, transcript_file, None)

    transcript_ext = transcript_file.split(".")[-1]

    transcript, *_ = utils.TranscriptReader(
        file_path=None, transcript_string=transcript_string, ext=transcript_ext
    ).read()

    empty_transcript = 0
    if len(transcript.keys()) == 0:
        empty_transcript += 1
        with open(f"{log_dir}/empty_transcripts.txt", "a") as f:
            f.write(f"{video_id}\n")
        return None, empty_transcript

    result = chunk_data(
        transcript=transcript,
        transcript_ext=transcript_ext,
        audio_file=audio_file,
        video_id=video_id,
        language=language,
        audio_only=audio_only,
        in_memory=in_memory,
        on_gcs=True,
        log_dir=log_dir,
    )

    return result


def chunk_transcript_only(
    transcript_data: Dict,
    transcript_manifest: Optional[List[str]],
    output_dir: str,
    dolma_format: bool = False,
    merge_man_mach: bool = False,
    in_memory: bool = True,
):
    transcript_string = transcript_data["content"]
    transcript_file = transcript_data["subtitle_file"]
    video_id = transcript_data["id"]
    language = None  # temporarily

    if len(transcript_data) > 6:
        keys_to_keep = list(transcript_data.keys())[6:]
    else:
        keys_to_keep = None

    segment_output_dir = os.path.dirname(transcript_file)
    transcript_ext = transcript_file.split(".")[-1]

    transcript, *_ = utils.TranscriptReader(
        file_path=None, transcript_string=transcript_string, ext=transcript_ext
    ).read()

    empty_transcript = 0
    if len(transcript.keys()) == 0:
        empty_transcript += 1
        return None, empty_transcript

    result = chunk_data(
        transcript=transcript,
        transcript_ext=transcript_ext,
        segment_output_dir=segment_output_dir,
        video_id=video_id,
        language=language,
        transcript_only=True,
        dolma_format=dolma_format,
        in_memory=in_memory,
    )

    segments_list, *counts = result
 
    if segments_list is not None:
        if merge_man_mach is False:
            segments_list = [
                segment
                for segment in segments_list
                if "/".join(
                    segment["audio_file"].split("/")[-2:]
                    if dolma_format is False
                    else segment["metadata"]["audio_file"].split("/")[-2:]
                )
                in transcript_manifest
            ]
        else:
            segments_list = [
                (
                    {**segment, "in_manifest": True}
                    if "/".join(
                        segment["audio_file"].split("/")[-2:]
                        if dolma_format is False
                        else segment["metadata"]["audio_file"].split("/")[-2:]
                    )
                    in transcript_manifest
                    else {**segment, "in_manifest": False}
                )
                for segment in segments_list
            ]

        if keys_to_keep is not None:
            for segment in segments_list:
                for key in keys_to_keep:
                    segment[key] = transcript_data[key]

        result = (segments_list, *counts)

    return result


def chunk_mach_transcript(
    transcript_data: Dict,
    log_dir: str,
    man_timestamps: Optional[List] = None,
    in_memory: bool = True,
    on_gcs: bool = False,
) -> Optional[List[Tuple[str, str, str, np.ndarray]]]:
    """Segment audio and transcript files into <= 30-second chunks

    Segment audio and transcript files into <= 30-second chunks. The audio and transcript files are represented by audio_file and transcript_file respectively.

    Args:
    transcript_file: Path to the transcript file
    audio_file: Path to the audio file

    Raises:
        Exception: If an error occurs during the chunking process
    """
    failed_transcript_count = 0

    try:
        transcript_string = transcript_data["mach_content"]
        transcript_file = transcript_data["subtitle_file"]
        video_id = transcript_data["id"]

        output_dir = os.path.dirname(transcript_file)
        get_ext = lambda transcript_string: (
            "vtt" if transcript_string.startswith("WEBVTT") else "srt"
        )
        transcript_ext = get_ext(transcript_string)

        transcript, *_ = utils.TranscriptReader(
            file_path=None, transcript_string=transcript_string, ext=transcript_ext
        ).read()

        empty_transcript = 0
        if len(transcript.keys()) == 0:
            empty_transcript += 1
            return None, empty_transcript

        a = 0
        b = 0

        segment_count = 0
        over_30_line_segment_count = 0
        bad_text_segment_count = 0
        over_ctx_len_segment_count = 0
        faulty_audio_segment_count = 0
        faulty_transcript_count = 0

        timestamps = list(transcript.keys())
        man_seg_idx = 0
        max_man_mach_diff = np.inf
        max_start_man_mach_diff = np.inf
        segments_list = []

        # to determine where to start
        while True:
            start_man_mach_diff = np.absolute(
                utils.convert_to_milliseconds(man_timestamps[man_seg_idx][0])
                - utils.convert_to_milliseconds(timestamps[a][0])
            )
            if start_man_mach_diff < max_start_man_mach_diff:
                max_start_man_mach_diff = start_man_mach_diff
                a += 1
            else:
                break

        a = a - 1
        b = a
        while (
            a < len(transcript) + 1
            and segment_count < SEGMENT_COUNT_THRESHOLD
            and man_seg_idx < len(man_timestamps)
        ):
            man_mach_diff = np.absolute(
                utils.convert_to_milliseconds(man_timestamps[man_seg_idx][1])
                - utils.convert_to_milliseconds(timestamps[b][1])
            )
            if man_mach_diff <= max_man_mach_diff:
                max_man_mach_diff = man_mach_diff
                b += 1
            elif man_mach_diff > max_man_mach_diff:
                # edge case (when transcript line is > 30s)
                if b == a:
                    over_30_line_segment_count += 1

                    if on_gcs:
                        with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")

                    a += 1
                    b += 1

                    if a == b == len(transcript):
                        if segment_count == 0:
                            faulty_transcript_count += 1
                        break

                    continue

                over_ctx_len, err = utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript, language=None
                )
                if not over_ctx_len:
                    t_output_file, transcript_string, _ = utils.write_segment(
                        timestamps=timestamps[a:b],
                        transcript=transcript,
                        output_dir=output_dir,
                        ext=transcript_ext,
                        in_memory=in_memory,
                    )

                    if not utils.too_short_audio_text(
                        start=timestamps[a][0], end=timestamps[b - 1][1]
                    ):
                        timestamp = t_output_file.split("/")[-1].split(
                            f".{transcript_ext}"
                        )[0]
                        segment = {
                            "subtitle_file": t_output_file.replace("ow_full", "ow_seg"),
                            "seg_content": transcript_string,
                            "timestamp": timestamp,
                            "id": video_id,
                            "audio_file": t_output_file.replace(
                                f".{transcript_ext}", ".npy"
                            ).replace("ow_full", "ow_seg"),
                        }
                        segments_list.append(segment)
                        segment_count += 1
                        man_seg_idx += 1
                else:
                    if err is not None:
                        if on_gcs:
                            with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                        bad_text_segment_count += 1
                    else:
                        if on_gcs:
                            with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                        over_ctx_len_segment_count += 1

                max_man_mach_diff = np.inf
                max_start_man_mach_diff = np.inf
                a = b
                if man_seg_idx < len(man_timestamps):
                    while True:
                        start_man_mach_diff = np.absolute(
                            utils.convert_to_milliseconds(
                                man_timestamps[man_seg_idx][0]
                            )
                            - utils.convert_to_milliseconds(timestamps[a][0])
                        )
                        if start_man_mach_diff < max_start_man_mach_diff:
                            max_start_man_mach_diff = start_man_mach_diff
                            a += 1
                        else:
                            break

                    a = a - 1
                    b = a

            if b == len(transcript):
                over_ctx_len, err = utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript, language=None
                )
                if not over_ctx_len:
                    t_output_file, transcript_string, _ = utils.write_segment(
                        timestamps=timestamps[a:b],
                        transcript=transcript,
                        output_dir=output_dir,
                        ext=transcript_ext,
                        in_memory=in_memory,
                    )

                    if not utils.too_short_audio_text(
                        start=timestamps[a][0], end=timestamps[b - 1][1]
                    ):
                        timestamp = t_output_file.split("/")[-1].split(
                            f".{transcript_ext}"
                        )[0]
                        segment = {
                            "subtitle_file": t_output_file.replace("ow_full", "ow_seg"),
                            "seg_content": transcript_string,
                            "timestamp": timestamp,
                            "id": video_id,
                            "audio_file": t_output_file.replace(
                                f".{transcript_ext}", ".npy"
                            ).replace("ow_full", "ow_seg"),
                        }
                        segments_list.append(segment)
                        segment_count += 1
                        man_seg_idx += 1
                else:
                    if err is not None:
                        if on_gcs:
                            with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                        bad_text_segment_count += 1
                    else:
                        if on_gcs:
                            with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                        over_ctx_len_segment_count += 1

                break
        if len(segments_list) == 0:
            return (
                None,
                over_30_line_segment_count,
                bad_text_segment_count,
                over_ctx_len_segment_count,
                faulty_audio_segment_count,
                faulty_transcript_count,
                failed_transcript_count
            )

        return (
            segments_list,
            over_30_line_segment_count,
            bad_text_segment_count,
            over_ctx_len_segment_count,
            faulty_audio_segment_count,
            faulty_transcript_count,
            failed_transcript_count
        )
    except ValueError as e:
        failed_transcript_count += 1
        if on_gcs:
            with open(f"{log_dir}/failed_chunking.txt", "a") as f:
                f.write(f"{video_id}\t{e}\n")
        return None, 0, 0, 0, 0, 0, failed_transcript_count
    except Exception as e:
        failed_transcript_count += 1
        if on_gcs:
            with open(f"{log_dir}/failed_chunking.txt", "a") as f:
                f.write(f"{video_id}\t{e}\n")
        return None, 0, 0, 0, 0, 0, failed_transcript_count


# deprecated
def chunk_audio_transcript(
    transcript_file: str,
    audio_file: str,
    output_dir: str,
    log_dir: str,
    failed_dir: str,
    in_memory: bool,
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
    if not in_memory:
        os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    try:
        # with open(f"/weka/huongn/processed_{transcript_file.split('/')[-3]}.txt", "a") as f:
        #     f.write(f"Processing {video_id_dir}\n")
        segment_output_dir = os.path.join(output_dir, transcript_file.split("/")[-2])
        if not in_memory:
            os.makedirs(segment_output_dir, exist_ok=True)
        transcript_ext = transcript_file.split(".")[-1]
        segment_count = 0

        transcript, *_ = utils.TranscriptReader(
            file_path=transcript_file, transcript_string=None, ext=transcript_ext
        ).read()

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

                    if audio_arr is not None and not utils.too_short_audio(
                        audio_arr=audio_arr
                    ):
                        segments_list.append(
                            (t_output_file, transcript_string, a_output_file, audio_arr)
                        )
                        segment_count += 1
                else:
                    with open(os.path.join(log_dir, "over_ctx_len.txt"), "a") as f:
                        f.write(f"{video_id_dir.split('/')[-1]}\tindex: {b}\n")

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

                    if audio_arr is not None and not utils.too_short_audio(
                        audio_arr=audio_arr
                    ):
                        segments_list.append(
                            (t_output_file, transcript_string, a_output_file, audio_arr)
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

                    if audio_arr is not None and not utils.too_short_audio(
                        audio_arr=audio_arr
                    ):
                        segments_list.append(
                            (t_output_file, transcript_string, a_output_file, audio_arr)
                        )
                        segment_count += 1
                else:
                    with open(os.path.join(log_dir, "over_ctx_len.txt"), "a") as f:
                        f.write(f"{video_id_dir.split('/')[-1]}\tindex: {b}\n")

                break
        if len(segments_list) == 0:
            return None
        # with open(f"/weka/huongn/processed_{transcript_file.split('/')[-3]}.txt", "a") as f:
        #     f.write(f"Processed {video_id_dir}\n")
        return segments_list
    except ValueError as e:
        with open(os.path.join(log_dir, "failed_chunking.txt"), "a") as f:
            f.write(f"{transcript_file}\t{audio_file}\t{e}\n")
        if not os.path.exists(failed_dir + "/" + video_id_dir.split("/")[-1]):
            shutil.move(video_id_dir, failed_dir)
        return None
    except Exception as e:
        with open(os.path.join(log_dir, "failed_chunking.txt"), "a") as f:
            f.write(f"{transcript_file}\t{audio_file}\t{e}\n")
        if not os.path.exists(failed_dir + "/" + video_id_dir.split("/")[-1]):
            shutil.move(video_id_dir, failed_dir)
        return None


def parallel_chunk_audio_transcript(
    args,
) -> Optional[List[Tuple[str, str, str, np.ndarray]]]:
    """Parallelized version of chunk_audio_transcript function to work in multiprocessing context"""
    return chunk_audio_transcript(*args)
