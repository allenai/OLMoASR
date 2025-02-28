import os
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
import time
import multiprocessing
from fire import Fire
from itertools import repeat, chain
import numpy as np
import sys
import glob
import json
import gzip
from whisper.normalizers import EnglishTextNormalizer
import jiwer
import Levenshtein
from collections import deque
import webvtt


sys.path.append(os.getcwd())
import segment_jsonl_utils as utils
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.FileHandler("main.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)
normalizer = EnglishTextNormalizer()

SEGMENT_COUNT_THRESHOLD = 120


def unarchive_jsonl_gz(file_path, output_path=None):
    """
    Unarchive a .jsonl.gz file and optionally save the uncompressed content to an output file.

    Args:
        file_path (str): Path to the .jsonl.gz file.
        output_path (str, optional): Path to save the uncompressed .jsonl file. Defaults to None.

    Returns:
        list[dict]: A list of JSON objects parsed from the .jsonl file.
    """
    data = []
    with gzip.open(file_path, "rt", encoding="utf-8") as gz_file:
        data = [json.loads(line.strip()) for line in gz_file]

    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_file:
            for json_obj in data:
                out_file.write(json.dumps(json_obj) + "\n")

    return data


def chunk_transcript(
    transcript_data: Dict,
    transcript_manifest: Optional[List[str]],
    log_dir: str,
    keep_tokens: bool = False,
    dolma_format: bool = False,
    mach_transcript: bool = False,
    merge_man_mach: bool = False,
    in_memory: bool = True,
) -> Optional[List[Tuple[str, str, str, np.ndarray]]]:
    """Segment audio and transcript files into <= 30-second chunks

    Segment audio and transcript files into <= 30-second chunks. The audio and transcript files are represented by audio_file and transcript_file respectively.

    Args:
    transcript_file: Path to the transcript file
    audio_file: Path to the audio file

    Raises:
        Exception: If an error occurs during the chunking process
    """
    try:
        transcript_string = (
            transcript_data["content"]
            if not mach_transcript
            else transcript_data["mach_content"]
        )
        transcript_file = transcript_data["subtitle_file"]
        if transcript_file.startswith("/weka"):
            video_id = transcript_file.split("/")[5]
        else:
            video_id = transcript_file.split("/")[1]

        output_dir = os.path.dirname(transcript_file)
        get_ext = lambda transcript_string: (
            "vtt" if transcript_string.startswith("WEBVTT") else "srt"
        )
        transcript_ext = (
            transcript_file.split(".")[-1]
            if not mach_transcript
            else get_ext(transcript_string)
        )
        segment_count = 0

        transcript, *_ = utils.TranscriptReader(
            file_path=None, transcript_string=transcript_string, ext=transcript_ext
        ).read()

        if len(transcript.keys()) == 0:
            with open(f"{log_dir}/empty_transcripts.txt", "a") as f:
                f.write(f"{video_id}\n")
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
                    with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                        f.write(f"{video_id}\tindex: {b}\n")

                    a += 1
                    b += 1

                    if a == b == len(transcript):
                        if segment_count == 0:
                            with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                                f.write(f"{video_id}\tdelete\n")
                        break

                    continue

                over_ctx_len, res = utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript, language=None
                )
                if not over_ctx_len:
                    t_output_file, transcript_string = utils.write_segment(
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
                        if dolma_format is True:
                            segment = {
                                "id": f"{video_id}_{segment_count}",
                                "text": transcript_string,
                                "source": "OW",
                                "metadata": {
                                    "subtitle_file": t_output_file.replace(
                                        "ow_full", "ow_seg"
                                    ),
                                    "timestamp": timestamp,
                                    "audio_file": t_output_file.replace(
                                        f".{transcript_ext}", ".npy"
                                    ).replace("ow_full", "ow_seg"),
                                },
                            }
                        else:
                            segment = {
                                "subtitle_file": t_output_file.replace(
                                    "ow_full", "ow_seg"
                                ),
                                "seg_content": transcript_string,
                                "full_content": transcript_data["content"],
                                "timestamp": timestamp,
                                "id": video_id,
                                "audio_file": t_output_file.replace(
                                    f".{transcript_ext}", ".npy"
                                ).replace("ow_full", "ow_seg"),
                            }
                        if keep_tokens and res is not None:
                            segment["tokens"] = res
                        segments_list.append(segment)
                        segment_count += 1
                else:
                    if type(res) is not List or res is not None:
                        with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")
                    elif res is None:
                        with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")

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
                            output_dir=output_dir,
                            ext=transcript_ext,
                            in_memory=in_memory,
                        )

                    if not utils.too_short_audio_text(start=start, end=end):
                        timestamp = t_output_file.split("/")[-1].split(
                            f".{transcript_ext}"
                        )[0]
                        if dolma_format is True:
                            segment = {
                                "id": f"{video_id}_{segment_count}",
                                "text": transcript_string,
                                "source": "OW",
                                "metadata": {
                                    "subtitle_file": t_output_file.replace(
                                        "ow_full", "ow_seg"
                                    ),
                                    "timestamp": timestamp,
                                    "audio_file": t_output_file.replace(
                                        f".{transcript_ext}", ".npy"
                                    ).replace("ow_full", "ow_seg"),
                                },
                            }
                        else:
                            segment = {
                                "subtitle_file": t_output_file.replace(
                                    "ow_full", "ow_seg"
                                ),
                                "seg_content": transcript_string,
                                "full_content": transcript_data["content"],
                                "timestamp": timestamp,
                                "id": video_id,
                                "audio_file": t_output_file.replace(
                                    f".{transcript_ext}", ".npy"
                                ).replace("ow_full", "ow_seg"),
                            }
                        if keep_tokens:
                            segment["tokens"] = [
                                50257,
                                50362,
                                50361,
                                50256,
                            ]  # tokenizer.sot_sequence_including_notimestamps + tokenizer.no_speech + tokenizer.eot
                        segments_list.append(segment)
                        segment_count += 1
                a = b

            if b == len(transcript) and diff < 30000:
                over_ctx_len, res = utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript, language=None
                )
                if not over_ctx_len:
                    t_output_file, transcript_string = utils.write_segment(
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
                        if dolma_format is True:
                            segment = {
                                "id": f"{video_id}_{segment_count}",
                                "text": transcript_string,
                                "source": "OW",
                                "metadata": {
                                    "subtitle_file": t_output_file.replace(
                                        "ow_full", "ow_seg"
                                    ),
                                    "timestamp": timestamp,
                                    "audio_file": t_output_file.replace(
                                        f".{transcript_ext}", ".npy"
                                    ).replace("ow_full", "ow_seg"),
                                },
                            }
                        else:
                            segment = {
                                "subtitle_file": t_output_file.replace(
                                    "ow_full", "ow_seg"
                                ),
                                "seg_content": transcript_string,
                                "full_content": transcript_data["content"],
                                "timestamp": timestamp,
                                "id": video_id,
                                "audio_file": t_output_file.replace(
                                    f".{transcript_ext}", ".npy"
                                ).replace("ow_full", "ow_seg"),
                            }
                        if keep_tokens and res is not None:
                            segment["tokens"] = res
                        segments_list.append(segment)
                        segment_count += 1
                else:
                    if type(res) is not List or res is not None:
                        with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")
                    elif res is None:
                        with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")

                break
        if len(segments_list) == 0:
            return None

        if not mach_transcript:
            if not merge_man_mach:
                segments_list = [
                    segment
                    for segment in segments_list
                    if "/".join(
                        segment["subtitle_file"].split("/")[-2:]
                        if dolma_format is False
                        else segment["metadata"]["subtitle_file"].split("/")[-2:]
                    )
                    in transcript_manifest
                ]
            else:
                segments_list = [
                    (
                        {**segment, "in_manifest": True}
                        if "/".join(
                            segment["subtitle_file"].split("/")[-2:]
                            if dolma_format is False
                            else segment["metadata"]["subtitle_file"].split("/")[-2:]
                        )
                        in transcript_manifest
                        else {**segment, "in_manifest": False}
                    )
                    for segment in segments_list
                ]

        return segments_list
    except ValueError as e:
        with open(f"{log_dir}/failed_chunking.txt", "a") as f:
            f.write(f"{video_id}\t{e}\n")
        return None
    except Exception as e:
        with open(f"{log_dir}/failed_chunking.txt", "a") as f:
            f.write(f"{video_id}\t{e}\n")
        return None


def parallel_chunk_transcript(
    args,
) -> Optional[List[Tuple[str, str, str, np.ndarray]]]:
    """Parallelized version of chunk_audio_transcript function to work in multiprocessing context"""
    return chunk_transcript(*args)


def chunk_mach_transcript(
    transcript_data: Dict,
    log_dir: str,
    man_timestamps: Optional[List] = None,
    in_memory: bool = True,
) -> Optional[List[Tuple[str, str, str, np.ndarray]]]:
    """Segment audio and transcript files into <= 30-second chunks

    Segment audio and transcript files into <= 30-second chunks. The audio and transcript files are represented by audio_file and transcript_file respectively.

    Args:
    transcript_file: Path to the transcript file
    audio_file: Path to the audio file

    Raises:
        Exception: If an error occurs during the chunking process
    """
    try:
        transcript_string = transcript_data["mach_content"]
        transcript_file = transcript_data["subtitle_file"]
        if transcript_file.startswith("/weka"):
            video_id = transcript_file.split("/")[5]
        else:
            video_id = transcript_file.split("/")[1]

        output_dir = os.path.dirname(transcript_file)
        get_ext = lambda transcript_string: (
            "vtt" if transcript_string.startswith("WEBVTT") else "srt"
        )
        transcript_ext = get_ext(transcript_string)
        segment_count = 0

        transcript, *_ = utils.TranscriptReader(
            file_path=None, transcript_string=transcript_string, ext=transcript_ext
        ).read()

        if len(transcript.keys()) == 0:
            with open(f"{log_dir}/empty_transcripts.txt", "a") as f:
                f.write(f"{video_id}\n")
            return None

        a = 0
        b = 0

        timestamps = list(transcript.keys())
        diff = 0
        init_diff = 0
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
            init_diff = utils.calculate_difference(timestamps[a][0], timestamps[b][1])

            man_mach_diff = np.absolute(
                utils.convert_to_milliseconds(man_timestamps[man_seg_idx][1])
                - utils.convert_to_milliseconds(timestamps[b][1])
            )
            if man_mach_diff <= max_man_mach_diff:
                diff = init_diff
                max_man_mach_diff = man_mach_diff
                b += 1
            elif man_mach_diff > max_man_mach_diff:
                # edge case (when transcript line is > 30s)
                if b == a:
                    with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                        f.write(f"{video_id}\tindex: {b}\n")

                    a += 1
                    b += 1

                    if a == b == len(transcript):
                        if segment_count == 0:
                            with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                                f.write(f"{video_id}\tdelete\n")
                        break

                    continue

                over_ctx_len, res = utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript, language=None
                )
                if not over_ctx_len:
                    t_output_file, transcript_string = utils.write_segment(
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
                            "full_content": transcript_data["mach_content"],
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
                    if type(res) is not List or res is not None:
                        with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")
                    elif res is None:
                        with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")

                init_diff = 0
                diff = 0
                max_man_mach_diff = np.inf
                max_start_man_mach_diff = np.inf

                # checking for silence
                # if timestamps[b][0] > timestamps[b - 1][1]:
                #     silence_segments = (
                #         utils.calculate_difference(
                #             timestamps[b - 1][1], timestamps[b][0]
                #         )
                #         // 30000
                #     )

                #     for i in range(0, silence_segments + 1):
                #         start = utils.adjust_timestamp(
                #             timestamps[b - 1][1], (i * 30000)
                #         )

                #         if i == silence_segments:
                #             if start == timestamps[b][0]:
                #                 continue
                #             else:
                #                 end = timestamps[b][0]
                #         else:
                #             end = utils.adjust_timestamp(start, 30000)

                #         t_output_file, transcript_string = utils.write_segment(
                #             timestamps=[(start, end)],
                #             transcript=None,
                #             output_dir=output_dir,
                #             ext=transcript_ext,
                #             in_memory=in_memory,
                #         )

                #     if not utils.too_short_audio_text(start=start, end=end):
                #         timestamp = t_output_file.split("/")[-1].split(
                #             f".{transcript_ext}"
                #         )[0]
                #         segment = {
                #             "subtitle_file": t_output_file.replace("ow_full", "ow_seg"),
                #             "seg_content": transcript_string,
                #             "timestamp": timestamp,
                #             "id": video_id,
                #             "audio_file": t_output_file.replace(
                #                 f".{transcript_ext}", ".npy"
                #             ).replace("ow_full", "ow_seg"),
                #         }
                #         segments_list.append(segment)
                #         segment_count += 1
                #         man_seg_idx += 1
                a = b
                if man_seg_idx < len(man_timestamps):
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


            if b == len(transcript):
                over_ctx_len, res = utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript, language=None
                )
                if not over_ctx_len:
                    t_output_file, transcript_string = utils.write_segment(
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
                            "full_content": transcript_data["mach_content"],
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
                    if type(res) is not List or res is not None:
                        with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")
                    elif res is None:
                        with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")

                break
        if len(segments_list) == 0:
            return None

        return segments_list
    except ValueError as e:
        with open(f"{log_dir}/failed_chunking.txt", "a") as f:
            f.write(f"{video_id}\t{e}\n")
        return None
    except Exception as e:
        with open(f"{log_dir}/failed_chunking.txt", "a") as f:
            f.write(f"{video_id}\t{e}\n")
        return None


def get_seg_text(segment):
    reader = utils.TranscriptReader(
        file_path=None,
        transcript_string=segment["seg_content"],
        ext="vtt" if segment["seg_content"].startswith("WEBVTT") else "srt",
    )
    t_dict, *_ = reader.read()
    segment_text = reader.extract_text(t_dict)
    return segment_text


def get_mach_seg_text(mach_segment):
    content = webvtt.from_string(mach_segment["seg_content"])
    # print(mach_segment)
    # for i, p in enumerate(content):
    #     print(f"{i=}")
    #     print(f"{p=}")
    # print(f"{content[0]=}")
    # print(f"{content[-1]=}")
    # print(f"{content[1]=}")
    # print(f"{len(content)=}")
    modified_content = []
    if len(content) > 0:
        if len(content) > 1:
            if content[0].text == content[1].text:
                modified_content.append(content[0])
                start = 2
            else:
                start = 0
        elif len(content) == 1:
            start = 0

        for i in range(start, len(content)):
            caption = content[i]
            if "\n" not in caption.text:
                modified_content.append(caption)
            elif "\n" in caption.text and i == len(content) - 1:
                caption.text = caption.text.split("\n")[-1]
                modified_content.append(caption)

        mach_segment_text = " ".join([caption.text for caption in modified_content])
    else:
        mach_segment_text = ""
    # print(f"{mach_segment_text=}")
    return mach_segment_text


def merge_man_mach_segs(
    transcript, transcript_manifest, shard_log_dir, keep_tokens, dolma_format, in_memory
):
    segments = chunk_transcript(
        transcript_data=transcript,
        transcript_manifest=transcript_manifest,
        log_dir=shard_log_dir,
        keep_tokens=keep_tokens,
        dolma_format=dolma_format,
        mach_transcript=False,
        merge_man_mach=True,
        in_memory=in_memory,
    )

    if segments is not None:
        shard_mach_log_dir = shard_log_dir + "_mach"
        os.makedirs(shard_mach_log_dir, exist_ok=True)
        man_timestamps = [
            segment["timestamp"].split("_")
            for segment in segments
            if not (
                segment["seg_content"] == "" or segment["seg_content"] == "WEBVTT\n\n"
            )
        ]
        mach_segments = chunk_mach_transcript(
            transcript_data=transcript,
            log_dir=shard_mach_log_dir,
            man_timestamps=man_timestamps,
            in_memory=in_memory,
        )

        new_segments = []
        if mach_segments is None:
            for segment in segments:
                if segment["in_manifest"] is True:
                    segment["mach_seg_content"] = "None"
                    segment["mach_full_content"] = "None"
                    segment["seg_text"] = "None"
                    segment["mach_seg_text"] = "None"
                    segment["mach_timestamp"] = ""
                    segment["edit_dist"] = 0.0
                    del segment["in_manifest"]
                    new_segments.append(segment)
        else:
            mach_full_content = mach_segments[0]["full_content"]
            mach_segments = deque(mach_segments)
            for segment in segments:
                seg_text = get_seg_text(segment)
                if seg_text != "":
                    if len(mach_segments) == 0:
                        norm_seg_text = normalizer(seg_text)
                        segment["seg_text"] = norm_seg_text
                        segment["mach_seg_text"] = ""
                        segment["mach_seg_content"] = ""
                        segment["mach_full_content"] = mach_full_content
                        segment["mach_timestamp"] = ""
                        # edit_dist = Levenshtein.distance(norm_seg_text, "")
                        if norm_seg_text != "":
                            edit_dist = jiwer.wer(norm_seg_text, "")
                        else:
                            edit_dist = jiwer.wer(seg_text, "")
                        segment["edit_dist"] = edit_dist
                        new_segments.append(segment)
                    else:
                        mach_segment = mach_segments.popleft()
                        if mach_segment["seg_content"] == "WEBVTT\n\n":
                            while mach_segment["seg_content"] == "WEBVTT\n\n":
                                if len(mach_segments) == 0:
                                    mach_segment = None
                                    break
                                else:
                                    mach_segment = mach_segments.popleft()
                        if segment["in_manifest"] is True:
                            mach_seg_text = get_mach_seg_text(mach_segment)
                            norm_mach_seg_text = normalizer(mach_seg_text)
                            norm_seg_text = normalizer(seg_text)
                            if norm_seg_text != "":
                                segment["seg_text"] = norm_seg_text
                                # edit_dist = Levenshtein.distance(norm_seg_text, norm_mach_seg_text)
                                edit_dist = jiwer.wer(norm_seg_text, norm_mach_seg_text)
                            else:
                                segment["seg_text"] = seg_text
                                # edit_dist = Levenshtein.distance(seg_text, norm_mach_seg_text)
                                edit_dist = jiwer.wer(seg_text, norm_mach_seg_text)
                            segment["mach_seg_text"] = norm_mach_seg_text
                            segment["mach_seg_content"] = mach_segment["seg_content"]
                            segment["mach_full_content"] = mach_full_content
                            segment["mach_timestamp"] = mach_segment["timestamp"]
                            segment["edit_dist"] = edit_dist
                            new_segments.append(segment)
                elif seg_text == "":
                    segment["seg_text"] = ""
                    segment["mach_seg_text"] = ""
                    segment["mach_seg_content"] = ""
                    segment["mach_full_content"] = mach_full_content
                    segment["mach_timestamp"] = ""
                    segment["edit_dist"] = 0.0
                    new_segments.append(segment)

                del segment["in_manifest"]

            if len(mach_segments) > 0:
                full_content = segments[0]["full_content"]
                for mach_segment in mach_segments:
                    mach_seg_text = normalizer(mach_segment["seg_content"])
                    new_segments.append(
                        {
                            "subtitle_file": "",
                            "seg_content": "",
                            "full_content": full_content,
                            "timestamp": "",
                            "id": segments[0]["id"],
                            "audio_file": "",
                            "seg_text": "",
                            "mach_seg_text": (
                                mach_seg_text
                                if mach_seg_text != ""
                                else mach_segment["seg_content"]
                            ),
                            "mach_seg_content": mach_segment["seg_content"],
                            "mach_full_content": mach_full_content,
                            "mach_timestamp": mach_segment["timestamp"],
                            "edit_dist": jiwer.wer(
                                (
                                    mach_seg_text
                                    if mach_seg_text != ""
                                    else mach_segment["seg_content"]
                                ),
                                "",
                            ),
                            # "edit_dist": Levenshtein.distance(
                            #     normalizer(mach_segment["seg_content"]), normalizer("")
                            # ),
                        }
                    )

        segments = new_segments

        return segments
    else:
        return None


def preprocess_jsonl(
    json_file: str,
    shard: str,
    transcript_manifest_file: str,
    log_dir: str,
    output_dir: str,
    only_subsample: bool = False,
    subsample: bool = False,
    subsample_size: int = 0,
    subsample_seed: int = 42,
    keep_tokens: bool = False,
    dolma_format: bool = False,
    seg_mach: bool = False,
    in_memory: bool = True,
):
    output_path = f"{output_dir}/shard_seg_{shard}.jsonl.gz"
    if os.path.exists(output_path):
        return 0, 0
    else:
        if not only_subsample:
            shard_log_dir = os.path.join(log_dir, shard)
            os.makedirs(shard_log_dir, exist_ok=True)

            if json_file.endswith(".gz"):
                transcript_data = unarchive_jsonl_gz(json_file)
            else:
                with open(json_file, "r") as f:
                    transcript_data = [json.loads(line.strip()) for line in f]

            with open(transcript_manifest_file, "r") as f:
                transcript_manifest = [line.strip() for line in f]

            if not seg_mach:
                segments_group = [
                    chunk_transcript(
                        transcript,
                        transcript_manifest,
                        shard_log_dir,
                        keep_tokens,
                        dolma_format,
                        in_memory,
                    )
                    for transcript in transcript_data
                ]
            else:
                segments_group = [
                    merge_man_mach_segs(
                        transcript=transcript,
                        transcript_manifest=transcript_manifest,
                        shard_log_dir=shard_log_dir,
                        keep_tokens=keep_tokens,
                        dolma_format=dolma_format,
                        in_memory=in_memory,
                    )
                    for transcript in transcript_data
                ]

            segments_group = [group for group in segments_group if group is not None]

            # Write the data to tar files
            segments_list = list(chain(*segments_group))
            seg_count = len(segments_list)
        else:
            with gzip.open(json_file, "rt", encoding="utf-8") as f:
                segments_list = [json.loads(line.strip()) for line in f]
            seg_count = len(segments_list)

        if subsample:
            if len(segments_list) > subsample_size:
                rng = np.random.default_rng(subsample_seed)
                subsampled_segments_list = rng.choice(
                    segments_list, size=subsample_size, replace=False
                )
                segments_list = subsampled_segments_list
                subsampled_count = len(subsampled_segments_list)
            else:
                subsampled_count = seg_count
                with open(
                    f"{log_dir}/less_than_subsample_size_{subsample_size}.txt", "a"
                ) as f:
                    f.write(
                        f"{shard} has less segments ({len(segments_list)}) than subsample size {subsample_size}"
                    )

            with gzip.open(output_path, "wt", encoding="utf-8") as f:
                for segment in segments_list:
                    f.write(json.dumps(segment) + "\n")
        else:
            with gzip.open(output_path, "wt", encoding="utf-8") as f:
                for segment in segments_list:
                    f.write(json.dumps(segment) + "\n")

        # return output_path, shard_log_dir, len(segments_list)
        return seg_count, subsampled_count if subsample else 0


def parallel_preprocess_jsonl(args):
    return preprocess_jsonl(*args)


def main(
    source_dir: str,
    manifest_dir: Optional[str],
    log_dir: str,
    output_dir: str,
    only_subsample: bool = False,
    subsample: bool = False,
    subsample_size: int = 0,
    subsample_seed: int = 42,
    keep_tokens: bool = False,
    dolma_format: bool = False,
    seg_mach: bool = False,
    in_memory: bool = True,
):
    print(
        f"{dolma_format=}, {only_subsample=}, {subsample=}, {subsample_size=}, {subsample_seed=}, {keep_tokens=}, {in_memory=}"
    )
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    shard_jsonls = glob.glob(f"{source_dir}/*.jsonl.gz")
    get_shard = lambda shard_jsonl: shard_jsonl.split("_")[-1].split(".")[0]

    shards = [get_shard(shard_jsonl) for shard_jsonl in shard_jsonls]
    logger.info(f"{len(shards)} shards found")
    logger.info(f"{shards[:5]=}")
    manifest_files = [f"{manifest_dir}/{shard}.txt" for shard in shards]
    logger.info(f"{manifest_files[:5]=}")
    with multiprocessing.Pool() as pool:
        counts = list(
            tqdm(
                pool.imap_unordered(
                    parallel_preprocess_jsonl,
                    zip(
                        shard_jsonls,
                        shards,
                        manifest_files,
                        repeat(log_dir),
                        repeat(output_dir),
                        repeat(only_subsample),
                        repeat(subsample),
                        repeat(subsample_size),
                        repeat(subsample_seed),
                        repeat(keep_tokens),
                        repeat(dolma_format),
                        repeat(seg_mach),
                        repeat(in_memory),
                    ),
                ),
                total=len(shard_jsonls),
            )
        )

    segment_counts, subsampled_counts = zip(*counts)

    logger.info(
        f"Total segment count: {sum(segment_counts)}, total duration: {(sum(segment_counts) * 30) / (60 * 60)} hours"
    )
    if subsample:
        logger.info(
            f"Total subsampled segment count: {sum(subsampled_counts)}, total duration: {(sum(subsampled_counts) * 30) / (60 * 60)} hours"
        )


if __name__ == "__main__":
    Fire(main)
