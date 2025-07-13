import os
from typing import Optional, List, Tuple, Dict, Union, Any
import numpy as np
import gzip
import json
import glob
from collections import deque
from itertools import repeat, chain
from tqdm import tqdm
import multiprocessing
from olmoasr import utils
from olmoasr.utils import (
    Segment,
    MachineSegment,
    SegmentCounter,
    read_file_in_tar,
    sum_counters,
    get_seg_text,
    get_mach_seg_text,
    unarchive_jsonl_gz,
)
import jiwer
from whisper.normalizers import EnglishTextNormalizer
from fire import Fire

normalizer = EnglishTextNormalizer()
SEGMENT_COUNT_THRESHOLD = 120


def chunk_data(
    transcript: Dict[Tuple[str, str], str],
    transcript_ext: str,
    audio_file: Optional[str] = None,
    segment_output_dir: Optional[str] = None,
    video_id: Optional[str] = None,
    language: Optional[str] = None,
    audio_only: bool = False,
    transcript_only: bool = False,
    in_memory: bool = True,
    on_gcs: bool = False,
    log_dir: Optional[str] = None,
) -> Tuple[Optional[List[Union[Segment, Tuple[str, Any]]]], SegmentCounter]:
    """
    Segment audio and transcript data into chunks of up to 30 seconds.

    This function takes a transcript dictionary and segments it into chunks that are
    approximately 30 seconds long. It handles speech and no-speech segments,
    validates text content, and can output transcript-only, audio-only, or both.

    Args:
        transcript: Dictionary mapping (start, end) timestamp tuples to transcript text
        transcript_ext: File extension for transcript files (e.g., 'srt', 'vtt')
        audio_file: Path to the audio file to segment
        segment_output_dir: Directory to save segmented output files
        video_id: Unique identifier for the video/audio content
        language: Language code for the transcript content
        audio_only: If True, only process audio segments
        transcript_only: If True, only process transcript segments
        in_memory: If True, keep segments in memory instead of writing to disk
        on_gcs: If True, running on Google Cloud Storage and will log to files
        log_dir: Directory for log files when on_gcs is True

    Returns:
        Tuple containing:
            - List of segment data (or None if no valid segments)
            - SegmentCounter object containing processing statistics

    Note:
        This function is intentionally complex as it handles multiple segment types,
        timing validations, and various output modes. Breaking it down further would
        require significant refactoring of the core segmentation logic.
    """
    # Initialize pointers and counters
    a = 0  # Starting pointer for current segment
    b = 0  # Ending pointer for current segment

    # Initialize counters for different types of issues/segments
    segment_counter = SegmentCounter()

    # Extract timestamps and initialize variables for segment processing
    timestamps = list(transcript.keys())
    diff = 0
    init_diff = 0
    segments_list = []
    from_no_speech = False
    start_in_no_speech = None
    local_start = None

    try:
        # Main segmentation loop: process transcript until reaching end or segment limit
        while (
            a < len(transcript) + 1
            and segment_counter.segment_count < SEGMENT_COUNT_THRESHOLD
        ):

            # === DETERMINE SEGMENT START POSITION ===
            # Set local_start based on whether we're starting fresh, after no-speech, or continuing
            if a == 0 and from_no_speech is False:
                if b == 1 and init_diff == 0:
                    local_start = timestamps[a][1]
                else:
                    local_start = timestamps[a][
                        0
                    ]  # starting from the beginning of the transcript
            elif from_no_speech is True or a == b:
                if start_in_no_speech is not None:
                    local_start = start_in_no_speech  # starting from no speech < 30s that is extracted from a > 30s no speech segment
                else:
                    local_start = timestamps[a][
                        0
                    ]  # starting immediately after no speech segment, when speech starts
            else:
                local_start = timestamps[a][
                    1
                ]  # starting immediately after previous segment speech ended

            # === CALCULATE SEGMENT DURATION ===
            # Calculate time difference between segment start and current end position
            init_diff = utils.calculate_difference(
                local_start,
                timestamps[b][1],
            )

            # === CHECK IF SEGMENT FITS WITHIN 30 SECONDS ===
            if init_diff <= 30000:  # 30 seconds = 30,000 milliseconds
                diff = init_diff
                b += 1  # Extend segment to include more content
            else:
                # Segment exceeds 30 seconds, need to process current segment

                # === HANDLE EDGE CASE: SINGLE LINE OVER 30 SECONDS ===
                if b == a:
                    segment_counter.over_30_line_segment_count += 1

                    if on_gcs and log_dir:
                        with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")

                    a += 1
                    b += 1
                    start_in_no_speech = None

                    # if reach end of transcript or transcript only has 1 line
                    if a == b == len(transcript):
                        # if transcript has only 1 line that is > 30s, stop processing
                        if segment_counter.segment_count == 0:
                            segment_counter.over_30_line_segment_count += 1
                        break

                    continue

                # === HANDLE NO-SPEECH SEGMENTS >=30 SECONDS ===
                # Check for consecutive no-speech periods that need special handling
                if (
                    b - a == 1
                    and local_start != timestamps[a][0]
                    and utils.calculate_difference(local_start, timestamps[b][0])
                    >= 30000
                    and (local_start, timestamps[b][0]) not in timestamps
                ):
                    # Calculate how many 30-second no-speech segments we can create
                    no_speech_segments = (
                        utils.calculate_difference(local_start, timestamps[b][0])
                        // 30000
                    )

                    # Create individual 30-second no-speech segments
                    for i in range(0, no_speech_segments + 1):
                        start = utils.adjust_timestamp(local_start, (i * 30000))

                        if i == no_speech_segments:
                            if start == timestamps[b][0]:
                                a = b
                                from_no_speech = True
                                start_in_no_speech = None
                                continue
                            else:
                                start_in_no_speech = start
                                from_no_speech = True
                                continue
                        else:
                            end = utils.adjust_timestamp(start, 30000)
                            norm_end = 30000
                            audio_timestamp = (
                                f"{start.replace('.', ',')}_{end.replace('.', ',')}"
                            )

                        # === PROCESS NO-SPEECH SEGMENT OUTPUT ===
                        if transcript_only is True:
                            result = utils.write_segment(
                                audio_begin=start,
                                timestamps=[(start, end)],
                                transcript=None,
                                output_dir=segment_output_dir,
                                ext=transcript_ext,
                                in_memory=in_memory,
                            )
                            if result is not None:
                                t_output_file, transcript_string = result[:2]
                            else:
                                continue

                            if not utils.too_short_audio_text(start=start, end=end):
                                timestamp = t_output_file.split("/")[-1].split(
                                    f".{transcript_ext}"
                                )[0]
                                outputs = Segment(
                                    subtitle_file=t_output_file,
                                    seg_content=transcript_string,
                                    text_timestamp=timestamp,
                                    audio_timestamp=audio_timestamp,
                                    norm_end=norm_end,
                                    video_id=video_id or "",
                                    seg_id=f"{video_id}_{segment_counter.segment_count}",
                                    audio_file=os.path.join(
                                        os.path.dirname(t_output_file),
                                        f"{audio_timestamp}.npy",
                                    ),
                                    ts_mode=True,
                                    no_ts_mode=True,
                                    only_no_ts_mode=False,
                                    num_tokens_no_ts_mode=0,
                                    num_tokens_ts_mode=0,
                                )
                                segments_list.append(outputs)
                                segment_counter.segment_count += 1
                        elif audio_only is True:
                            result = utils.trim_audio(
                                audio_file=audio_file,
                                start=start,
                                end=end,  # like speech segments
                                output_dir=segment_output_dir,
                                in_memory=in_memory,
                            )
                            if result is not None:
                                a_output_file, audio_arr = result
                            else:
                                continue
                        elif transcript_only is False and audio_only is False:
                            t_result = utils.write_segment(
                                audio_begin=start,
                                timestamps=[(start, end)],
                                transcript=None,
                                output_dir=segment_output_dir,
                                ext=transcript_ext,
                                in_memory=in_memory,
                            )
                            if t_result is not None:
                                t_output_file, transcript_string = t_result[:2]
                            else:
                                continue

                            a_result = utils.trim_audio(
                                audio_file=audio_file,
                                start=start,
                                end=end,  # like speech segments
                                output_dir=segment_output_dir,
                                in_memory=in_memory,
                            )
                            if a_result is not None:
                                a_output_file, audio_arr = a_result
                            else:
                                continue

                        # === VALIDATE AND STORE AUDIO SEGMENTS ===
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
                                segment_counter.segment_count += 1
                            else:
                                if audio_arr is None:
                                    if on_gcs and log_dir:
                                        with open(
                                            f"{log_dir}/faulty_audio.txt", "a"
                                        ) as f:
                                            f.write(f"{video_id}\tindex: {b}\n")
                                    else:
                                        segment_counter.faulty_audio_segment_count += 1
                    continue
                elif (
                    b - a == 1
                    and local_start != timestamps[a][0]
                    and utils.calculate_difference(local_start, timestamps[b][0])
                    < 30000
                    and (local_start, timestamps[b][0]) not in timestamps
                ):
                    if timestamps[b][0] == local_start:
                        a = b
                        from_no_speech = True
                        start_in_no_speech = None
                        continue
                    end = timestamps[b][0]

                    if utils.convert_to_milliseconds(
                        end
                    ) < utils.convert_to_milliseconds(local_start):
                        only_no_ts_mode = True
                    else:
                        only_no_ts_mode = False

                    norm_end = utils.adjust_timestamp(
                        end, -utils.convert_to_milliseconds(local_start)
                    )
                    audio_timestamp = f"{local_start.replace('.', ',')}_{utils.adjust_timestamp(local_start, 30000).replace('.', ',')}"

                    if transcript_only is True:
                        result = utils.write_segment(
                            audio_begin=local_start,
                            timestamps=[(local_start, end)],
                            transcript=None,
                            output_dir=segment_output_dir,
                            ext=transcript_ext,
                            in_memory=in_memory,
                        )
                        if result is not None:
                            t_output_file, transcript_string = result[:2]
                        else:
                            continue

                        if not utils.too_short_audio_text(start=local_start, end=end):
                            timestamp = t_output_file.split("/")[-1].split(
                                f".{transcript_ext}"
                            )[0]
                            outputs = Segment(
                                subtitle_file=t_output_file,
                                seg_content=transcript_string,
                                text_timestamp=timestamp,
                                audio_timestamp=audio_timestamp,
                                norm_end=norm_end,
                                video_id=video_id or "",
                                seg_id=f"{video_id}_{segment_counter.segment_count}",
                                audio_file=os.path.join(
                                    os.path.dirname(t_output_file),
                                    f"{audio_timestamp}.npy",
                                ),
                                ts_mode=True,
                                no_ts_mode=True,
                                only_no_ts_mode=only_no_ts_mode,
                                num_tokens_no_ts_mode=0,
                                num_tokens_ts_mode=0,
                            )
                            segments_list.append(outputs)
                            segment_counter.segment_count += 1
                    elif audio_only is True:
                        result = utils.trim_audio(
                            audio_file=audio_file,
                            start=local_start,
                            end=utils.adjust_timestamp(
                                local_start, 30000
                            ),  # like speech segments
                            output_dir=segment_output_dir,
                            in_memory=in_memory,
                        )
                        if result is not None:
                            a_output_file, audio_arr = result
                        else:
                            continue
                    elif transcript_only is False and audio_only is False:
                        t_result = utils.write_segment(
                            audio_begin=local_start,
                            timestamps=[(local_start, end)],
                            transcript=None,
                            output_dir=segment_output_dir,
                            ext=transcript_ext,
                            in_memory=in_memory,
                        )
                        if t_result is not None:
                            t_output_file, transcript_string = t_result[:2]
                        else:
                            continue

                        a_result = utils.trim_audio(
                            audio_file=audio_file,
                            start=local_start,
                            end=utils.adjust_timestamp(
                                local_start, 30000
                            ),  # like speech segments
                            output_dir=segment_output_dir,
                            in_memory=in_memory,
                        )
                        if a_result is not None:
                            a_output_file, audio_arr = a_result
                        else:
                            continue

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
                            segment_counter.segment_count += 1
                        else:
                            if audio_arr is None:
                                if on_gcs and log_dir:
                                    with open(f"{log_dir}/faulty_audio.txt", "a") as f:
                                        f.write(f"{video_id}\tindex: {b}\n")
                                else:
                                    segment_counter.faulty_audio_segment_count += 1
                    a = b
                    from_no_speech = True
                    start_in_no_speech = None
                    continue
                elif (
                    b - a == 1
                    and local_start != timestamps[a][0]
                    and (
                        utils.calculate_difference(local_start, timestamps[b][0])
                        < 30000
                        or utils.calculate_difference(local_start, timestamps[b][0])
                        >= 30000
                    )
                    and (local_start, timestamps[b][0]) in timestamps
                ):
                    a = b
                    continue

                # a + 1 is the beginning of the text line ([a + 1][0] can be == a[-1] in terms of timestamps, but we don't want to include text in a b/c a == b - 1. [a + 1][0] can also be == a[-1] )
                over_ctx_len, res = utils.over_ctx_len(
                    timestamps=(
                        timestamps[a:b]  # directly starting when speech starts
                        if a == 0
                        or (
                            start_in_no_speech is None
                            and a > 0
                            and from_no_speech is True
                        )
                        else timestamps[a + 1 : b]  # starting when there's no speech
                    ),
                    transcript=transcript,
                    language=language,
                )
                valid_timestamps = utils.timestamps_valid(
                    timestamps=(
                        timestamps[a:b]
                        if a == 0
                        or (
                            start_in_no_speech is None
                            and a > 0
                            and from_no_speech is True
                        )
                        else timestamps[a + 1 : b]
                    ),
                    global_start=timestamps[0][0],
                    global_end=timestamps[-1][1],
                )
                # check if segment text goes over model context length
                if not over_ctx_len and valid_timestamps:
                    if transcript_only is True:
                        # writing transcript segment w/ timestamps[a + 1][0] -> timestamps[b - 1][1]
                        result = utils.write_segment(
                            audio_begin=local_start,
                            timestamps=(
                                timestamps[a:b]
                                if a == 0
                                or (
                                    start_in_no_speech is None
                                    and a > 0
                                    and from_no_speech is True
                                )
                                else timestamps[a + 1 : b]
                            ),
                            transcript=transcript,
                            output_dir=segment_output_dir,
                            ext=transcript_ext,
                            in_memory=in_memory,
                        )
                        if result is not None and len(result) >= 4:
                            (
                                t_output_file,
                                transcript_string,
                                norm_end,
                                only_no_ts_mode,
                            ) = result
                        else:
                            continue

                        if not utils.too_short_audio_text(
                            start=local_start,
                            end=utils.adjust_timestamp(local_start, 30000),
                        ):
                            audio_timestamp = f"{local_start.replace('.', ',')}_{utils.adjust_timestamp(local_start, 30000).replace('.', ',')}"
                            text_timestamp = t_output_file.split("/")[-1].split(
                                f".{transcript_ext}"
                            )[0]
                            outputs = Segment(
                                subtitle_file=t_output_file,
                                seg_content=transcript_string,
                                text_timestamp=text_timestamp,
                                audio_timestamp=audio_timestamp,
                                norm_end=norm_end,
                                video_id=video_id or "",
                                seg_id=f"{video_id}_{segment_counter.segment_count}",
                                audio_file=os.path.join(
                                    os.path.dirname(t_output_file),
                                    f"{audio_timestamp}.npy",
                                ),
                                ts_mode=(
                                    res.get("ts_mode", True)
                                    if isinstance(res, dict)
                                    else True
                                ),
                                no_ts_mode=(
                                    res.get("no_ts_mode", True)
                                    if isinstance(res, dict)
                                    else True
                                ),
                                only_no_ts_mode=only_no_ts_mode,
                                num_tokens_no_ts_mode=(
                                    res.get("num_tokens_no_ts_mode", 0)
                                    if isinstance(res, dict)
                                    else 0
                                ),
                                num_tokens_ts_mode=(
                                    res.get("num_tokens_ts_mode", 0)
                                    if isinstance(res, dict)
                                    else 0
                                ),
                            )
                            segments_list.append(outputs)
                            segment_counter.segment_count += 1
                    elif audio_only is True:
                        # writing audio segment w/ local_start -> local_start + 30s
                        result = utils.trim_audio(
                            audio_file=audio_file,
                            start=local_start,
                            end=utils.adjust_timestamp(local_start, 30000),
                            output_dir=segment_output_dir,
                            in_memory=in_memory,
                        )
                        if result is not None:
                            a_output_file, audio_arr = result
                        else:
                            continue
                    elif transcript_only is False and audio_only is False:
                        # writing transcript segment w/ timestamps[a + 1][0] or timestamps[a][0] -> timestamps[b - 1][1]
                        t_result = utils.write_segment(
                            audio_begin=local_start,
                            timestamps=(
                                timestamps[a:b]
                                if a == 0
                                or (
                                    start_in_no_speech is None
                                    and a > 0
                                    and from_no_speech is True
                                )
                                else timestamps[a + 1 : b]
                            ),
                            transcript=transcript,
                            output_dir=segment_output_dir,
                            ext=transcript_ext,
                            in_memory=in_memory,
                        )
                        if t_result is not None:
                            t_output_file, transcript_string = t_result[:2]
                        else:
                            continue

                        # writing audio segment w/ local_start -> local_start + 30s
                        a_result = utils.trim_audio(
                            audio_file=audio_file,
                            start=local_start,
                            end=utils.adjust_timestamp(local_start, 30000),
                            output_dir=segment_output_dir,
                            in_memory=in_memory,
                        )
                        if a_result is not None:
                            a_output_file, audio_arr = a_result
                        else:
                            continue

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
                            segment_counter.segment_count += 1
                        else:
                            if audio_arr is None:
                                if on_gcs and log_dir:
                                    with open(f"{log_dir}/faulty_audio.txt", "a") as f:
                                        f.write(f"{video_id}\tindex: {b}\n")
                                else:
                                    segment_counter.faulty_audio_segment_count += 1
                else:
                    if res == "error":
                        if on_gcs and log_dir:
                            with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                        segment_counter.bad_text_segment_count += 1
                    else:
                        if on_gcs and log_dir:
                            with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                        segment_counter.over_ctx_len_segment_count += 1

                init_diff = 0
                diff = 0
                # moving pointer a to b - 1
                a = b - 1

                # resetting
                from_no_speech = False
                start_in_no_speech = None

            # at the end of transcript
            if b == len(transcript) and diff <= 30000:
                over_ctx_len, res = utils.over_ctx_len(
                    timestamps=(
                        timestamps[a:b]
                        if a == 0
                        or (
                            start_in_no_speech is None
                            and a > 0
                            and from_no_speech is True
                        )
                        else timestamps[a + 1 : b]
                    ),
                    transcript=transcript,
                    language=language,
                    last_seg=True,
                )
                valid_timestamps = utils.timestamps_valid(
                    timestamps=(
                        timestamps[a:b]
                        if a == 0
                        or (
                            start_in_no_speech is None
                            and a > 0
                            and from_no_speech is True
                        )
                        else timestamps[a + 1 : b]
                    ),
                    global_start=timestamps[0][0],
                    global_end=timestamps[-1][1],
                )
                if not over_ctx_len and valid_timestamps:
                    if transcript_only is True:
                        result = utils.write_segment(
                            audio_begin=local_start,
                            timestamps=(
                                timestamps[a:b]
                                if a == 0
                                or (
                                    start_in_no_speech is None
                                    and a > 0
                                    and from_no_speech is True
                                )
                                else timestamps[a + 1 : b]
                            ),
                            transcript=transcript,
                            output_dir=segment_output_dir,
                            ext=transcript_ext,
                            in_memory=in_memory,
                        )
                        if result is not None and len(result) >= 4:
                            (
                                t_output_file,
                                transcript_string,
                                norm_end,
                                only_no_ts_mode,
                            ) = result
                        else:
                            continue

                        if not utils.too_short_audio_text(
                            start=local_start, end=timestamps[b - 1][1]
                        ):
                            timestamp = t_output_file.split("/")[-1].split(
                                f".{transcript_ext}"
                            )[0]
                            outputs = Segment(
                                subtitle_file=t_output_file,
                                seg_content=transcript_string,
                                text_timestamp=timestamp,
                                audio_timestamp=timestamp,
                                norm_end=norm_end,
                                video_id=video_id or "",
                                seg_id=f"{video_id}_{segment_counter.segment_count}",
                                audio_file=t_output_file.replace(
                                    f".{transcript_ext}", ".npy"
                                ),
                                ts_mode=(
                                    res.get("ts_mode", True)
                                    if isinstance(res, dict)
                                    else True
                                ),
                                no_ts_mode=(
                                    res.get("no_ts_mode", True)
                                    if isinstance(res, dict)
                                    else True
                                ),
                                only_no_ts_mode=only_no_ts_mode,
                                num_tokens_no_ts_mode=(
                                    res.get("num_tokens_no_ts_mode", 0)
                                    if isinstance(res, dict)
                                    else 0
                                ),
                                num_tokens_ts_mode=(
                                    res.get("num_tokens_ts_mode", 0)
                                    if isinstance(res, dict)
                                    else 0
                                ),
                            )
                            segments_list.append(outputs)
                            segment_counter.segment_count += 1
                    elif audio_only is True:
                        result = utils.trim_audio(
                            audio_file=audio_file,
                            start=local_start,
                            end=timestamps[b - 1][1],
                            output_dir=segment_output_dir,
                            in_memory=in_memory,
                        )
                        if result is not None:
                            a_output_file, audio_arr = result
                        else:
                            continue
                    elif transcript_only is False and audio_only is False:
                        t_result = utils.write_segment(
                            audio_begin=local_start,
                            timestamps=(
                                timestamps[a:b]
                                if a == 0
                                or (
                                    start_in_no_speech is None
                                    and a > 0
                                    and from_no_speech is True
                                )
                                else timestamps[a + 1 : b]
                            ),
                            transcript=transcript,
                            output_dir=segment_output_dir,
                            ext=transcript_ext,
                            in_memory=in_memory,
                        )
                        if t_result is not None:
                            t_output_file, transcript_string = t_result[:2]
                        else:
                            continue

                        a_result = utils.trim_audio(
                            audio_file=audio_file,
                            start=local_start,
                            end=timestamps[b - 1][1],
                            output_dir=segment_output_dir,
                            in_memory=in_memory,
                        )
                        if a_result is not None:
                            a_output_file, audio_arr = a_result
                        else:
                            continue

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
                            segment_counter.segment_count += 1
                        else:
                            if audio_arr is None:
                                if on_gcs and log_dir:
                                    with open(f"{log_dir}/faulty_audio.txt", "a") as f:
                                        f.write(f"{video_id}\tindex: {b}\n")
                                else:
                                    segment_counter.faulty_audio_segment_count += 1
                else:
                    if res == "error":
                        if on_gcs and log_dir:
                            with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                        segment_counter.bad_text_segment_count += 1
                    else:
                        if on_gcs and log_dir:
                            with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                        segment_counter.over_ctx_len_segment_count += 1

                break

        if len(segments_list) == 0:
            return (None, segment_counter)

        return (segments_list, segment_counter)
    except ValueError as e:
        segment_counter.failed_transcript_count += 1
        if on_gcs and log_dir:
            with open(f"{log_dir}/failed_chunking.txt", "a") as f:
                f.write(f"{video_id}\t{e}\n")
        return None, segment_counter
    except Exception as e:
        segment_counter.failed_transcript_count += 1
        if on_gcs and log_dir:
            with open(f"{log_dir}/failed_chunking.txt", "a") as f:
                f.write(f"{video_id}\t{e}\n")
        return None, segment_counter


def chunk_local(
    transcript_file: str,
    audio_file: str,
    output_dir: str,
    audio_only: bool,
    transcript_only: bool,
    in_memory: bool,
) -> Tuple[
    Optional[Tuple[Optional[List[Union[Segment, Tuple[str, Any]]]], SegmentCounter]],
    int,
]:
    """
    Segment local audio and transcript files into chunks.

    Processes local files by reading the transcript and segmenting both audio and
    transcript content into manageable chunks for training or inference.

    Args:
        transcript_file: Path to the transcript file (SRT or VTT format)
        audio_file: Path to the audio file to segment
        output_dir: Directory to save the segmented output files
        audio_only: If True, only process audio segments
        transcript_only: If True, only process transcript segments
        in_memory: If True, keep segments in memory instead of writing to disk

    Returns:
        Tuple containing:
            - Result tuple from chunk_data function or None
            - Count of empty transcripts encountered
    """
    # Set up directory structure and extract file information
    language = None  # English-only (for now?)
    os.makedirs(output_dir, exist_ok=True)
    video_id = os.path.dirname(transcript_file).split("/")[-1]
    segment_output_dir = os.path.join(output_dir, video_id)
    os.makedirs(segment_output_dir, exist_ok=True)
    transcript_ext = transcript_file.split(".")[-1]

    # Read and parse the transcript file
    try:
        reader = utils.TranscriptReader(
            file_path=transcript_file, transcript_string=None, ext=transcript_ext
        )
        result = reader.read()
        if result is None:
            return None, 1
        transcript = result[0] if isinstance(result, tuple) else result
    except Exception:
        return None, 1

    # Check for empty transcript
    empty_transcript = 0
    if len(transcript.keys()) == 0:
        empty_transcript += 1
        return None, empty_transcript

    # Process the transcript through the main chunking function
    chunk_result = chunk_data(
        transcript=transcript,
        transcript_ext=transcript_ext,
        audio_file=audio_file,
        segment_output_dir=segment_output_dir,
        video_id=video_id,
        language=language,
        audio_only=audio_only,
        transcript_only=transcript_only,
        in_memory=in_memory,
    )

    return chunk_result, empty_transcript


def chunk_gcs(
    tar_gz_file: str,
    transcript_file: str,
    audio_file: str,
    log_dir: str,
    audio_only: bool = True,
    in_memory: bool = True,
) -> Tuple[
    Optional[Tuple[Optional[List[Union[Segment, Tuple[str, Any]]]], SegmentCounter]],
    int,
]:
    """
    Segment audio and transcript files from Google Cloud Storage tar.gz archives.

    Processes files stored in tar.gz archives on GCS by extracting transcript content
    and segmenting it into chunks. Logs processing errors to specified log directory.

    Args:
        tar_gz_file: Path to the tar.gz archive file
        transcript_file: Path to transcript file within the archive
        audio_file: Path to audio file within the archive
        log_dir: Directory to save log files for error tracking
        audio_only: If True, only process audio segments
        in_memory: If True, keep segments in memory instead of writing to disk

    Returns:
        Tuple containing:
            - Result tuple from chunk_data function or None
            - Count of empty transcripts encountered
    """
    # Extract video ID from the transcript file path structure
    video_id = transcript_file.split("/")[1]

    # Try to extract language from tar filename (if encoded in filename)
    if "_" in tar_gz_file:
        language = tar_gz_file.split("_")[-1].split(".")[0]
    else:
        language = None

    # Extract transcript content from the tar.gz archive
    transcript_string = read_file_in_tar(tar_gz_file, transcript_file, None)
    if transcript_string is None:
        return None, 1

    # Determine transcript file extension for processing
    transcript_ext = transcript_file.split(".")[-1]

    # Parse the transcript string into structured format
    try:
        reader = utils.TranscriptReader(
            file_path=None, transcript_string=transcript_string, ext=transcript_ext
        )
        result = reader.read()
        if result is None:
            return None, 1
        transcript = result[0] if isinstance(result, tuple) else result
    except Exception:
        return None, 1

    # Handle empty transcripts and log them
    empty_transcript = 0
    if len(transcript.keys()) == 0:
        empty_transcript += 1
        with open(f"{log_dir}/empty_transcripts.txt", "a") as f:
            f.write(f"{video_id}\n")
        return None, empty_transcript

    # Process the transcript through the main chunking function with GCS-specific settings
    chunk_result = chunk_data(
        transcript=transcript,
        transcript_ext=transcript_ext,
        audio_file=audio_file,
        video_id=video_id,
        language=language,
        audio_only=audio_only,
        in_memory=in_memory,
        on_gcs=True,  # Enable GCS-specific logging
        log_dir=log_dir,
    )

    return chunk_result, empty_transcript


def chunk_transcript_only(
    transcript_data: Dict[str, Any],
    in_memory: bool = True,
) -> Tuple[Optional[List[Segment]], Union[SegmentCounter, int]]:
    """
    Segment transcript data without corresponding audio files.

    Processes transcript-only data by chunking the text content into segments.
    Can filter segments based on manifest and format output for different use cases.

    Args:
        transcript_data: Dictionary containing transcript content and metadata
        in_memory: If True, keep segments in memory instead of writing to disk

    Returns:
        Tuple containing:
            - List of Segment objects (or None if no valid segments)
            - SegmentCounter object containing statistics about the chunking process
    """
    # Extract transcript information from input data
    transcript_string = transcript_data["content"]
    transcript_file = transcript_data["subtitle_file"]
    video_id = transcript_data["id"]
    language = None  # English-only (for now?)

    # Determine if there are additional metadata keys to preserve
    if len(transcript_data) > 6:
        keys_to_keep = list(transcript_data.keys())[6:]
    else:
        keys_to_keep = None

    # Set up output directory and file extension
    segment_output_dir = os.path.dirname(transcript_file)
    transcript_ext = transcript_file.split(".")[-1]

    # Parse the transcript string into structured format
    try:
        reader = utils.TranscriptReader(
            file_path=None, transcript_string=transcript_string, ext=transcript_ext
        )
        result = reader.read()
        if result is None:
            return None, 1
        transcript = result[0] if isinstance(result, tuple) else result
    except Exception:
        return None, 1

    # Check for empty transcript
    empty_transcript = 0
    if len(transcript.keys()) == 0:
        empty_transcript += 1
        return None, empty_transcript

    # Process transcript through chunking function (transcript-only mode)
    chunk_result = chunk_data(
        transcript=transcript,
        transcript_ext=transcript_ext,
        segment_output_dir=segment_output_dir,
        video_id=video_id,
        language=language,
        transcript_only=True,
        in_memory=in_memory,
    )

    segments_list, segment_counter = chunk_result

    # Post-process segments based on manifest and format requirements
    if segments_list is not None:
        # Filter to only include Segment objects (transcript-only mode should only return Segments)
        segment_objects = [seg for seg in segments_list if isinstance(seg, Segment)]

        # Preserve additional metadata keys from original transcript data
        if keys_to_keep is not None:
            for segment in segment_objects:
                for key in keys_to_keep:
                    segment.add_attr(key, transcript_data[key])

        return segment_objects, segment_counter

    return None, segment_counter


def chunk_mach_transcript(
    transcript_data: Dict[str, Any],
    log_dir: str,
    man_timestamps: Optional[List[Tuple[str, str]]] = None,
    in_memory: bool = True,
    on_gcs: bool = False,
) -> Tuple[Optional[List[MachineSegment]], Union[SegmentCounter, int]]:
    """
    Segment machine-generated transcripts aligned with manual transcript timestamps.

    Processes machine-generated transcript data by aligning it with manual transcript
    timestamps to create consistent segments. This is useful for training data where
    both manual and machine transcripts are available.

    Args:
        transcript_data: Dictionary containing machine transcript content and metadata
        log_dir: Directory to save log files for error tracking
        man_timestamps: List of manual transcript timestamps for alignment
        in_memory: If True, keep segments in memory instead of writing to disk
        on_gcs: If True, running on Google Cloud Storage and will log to files

    Returns:
        Tuple containing:
            - List of MachineSegment objects (or None if no valid segments)
            - SegmentCounter object containing statistics about the chunking process

    Raises:
        ValueError: If there's an error during transcript processing
        Exception: If any other error occurs during the chunking process
    """
    if man_timestamps is None:
        return None, 1

    try:
        transcript_string = transcript_data["mach_content"]
        transcript_file = transcript_data["subtitle_file"]
        video_id = transcript_data["id"]

        output_dir = os.path.dirname(transcript_file)
        get_ext = lambda transcript_string: (
            "vtt" if transcript_string.startswith("WEBVTT") else "srt"
        )
        transcript_ext = get_ext(transcript_string)

        try:
            reader = utils.TranscriptReader(
                file_path=None, transcript_string=transcript_string, ext=transcript_ext
            )
            result = reader.read()
            if result is None:
                return None, 1
            transcript = result[0] if isinstance(result, tuple) else result
        except Exception:
            return None, 1

        empty_transcript = 0
        if len(transcript.keys()) == 0:
            empty_transcript += 1
            return None, empty_transcript

        a = 0
        b = 0

        segment_counter = SegmentCounter()

        timestamps = list(transcript.keys())
        man_seg_idx = 0
        max_man_mach_diff = np.inf
        max_start_man_mach_diff = np.inf
        segments_list = []

        # to determine where to start
        while True:
            if man_seg_idx >= len(man_timestamps) or a >= len(timestamps):
                break
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
            and segment_counter.segment_count < SEGMENT_COUNT_THRESHOLD
            and man_seg_idx < len(man_timestamps)
        ):
            if b >= len(timestamps):
                break
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
                    segment_counter.over_30_line_segment_count += 1

                    if on_gcs:
                        with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")

                    a += 1
                    b += 1

                    if a == b == len(transcript):
                        if segment_counter.segment_count == 0:
                            segment_counter.over_30_line_segment_count += 1
                        break

                    continue

                over_ctx_len, res = utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript, language=None
                )
                if not over_ctx_len:
                    result = utils.write_segment(
                        audio_begin=timestamps[a][0],
                        timestamps=timestamps[a:b],
                        transcript=transcript,
                        output_dir=output_dir,
                        ext=transcript_ext,
                        in_memory=in_memory,
                    )
                    if result is not None:
                        t_output_file, transcript_string = result[:2]
                    else:
                        continue

                    if not utils.too_short_audio_text(
                        start=timestamps[a][0], end=timestamps[b - 1][1]
                    ):
                        timestamp = t_output_file.split("/")[-1].split(
                            f".{transcript_ext}"
                        )[0]
                        segment = MachineSegment(
                            subtitle_file=t_output_file,
                            seg_content=transcript_string,
                            timestamp=timestamp,
                            video_id=video_id,
                            audio_file=t_output_file.replace(
                                f".{transcript_ext}", ".npy"
                            ),
                        )
                        segments_list.append(segment)
                        segment_counter.segment_count += 1
                        man_seg_idx += 1
                else:
                    if res is not None:
                        if on_gcs:
                            with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                        segment_counter.bad_text_segment_count += 1
                    else:
                        if on_gcs:
                            with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                        segment_counter.over_ctx_len_segment_count += 1

                max_man_mach_diff = np.inf
                max_start_man_mach_diff = np.inf
                a = b
                if man_seg_idx < len(man_timestamps):
                    while True:
                        if a >= len(timestamps):
                            break
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
                over_ctx_len, res = utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript, language=None
                )
                if not over_ctx_len:
                    result = utils.write_segment(
                        audio_begin=timestamps[a][0],
                        timestamps=timestamps[a:b],
                        transcript=transcript,
                        output_dir=output_dir,
                        ext=transcript_ext,
                        in_memory=in_memory,
                    )
                    if result is not None:
                        t_output_file, transcript_string = result[:2]
                    else:
                        continue

                    if not utils.too_short_audio_text(
                        start=timestamps[a][0], end=timestamps[b - 1][1]
                    ):
                        timestamp = t_output_file.split("/")[-1].split(
                            f".{transcript_ext}"
                        )[0]
                        segment = MachineSegment(
                            subtitle_file=t_output_file,
                            seg_content=transcript_string,
                            timestamp=timestamp,
                            video_id=video_id,
                            audio_file=t_output_file.replace(
                                f".{transcript_ext}", ".npy"
                            ),
                        )
                        segments_list.append(segment)
                        segment_counter.segment_count += 1
                        man_seg_idx += 1
                else:
                    if res is not None:
                        if on_gcs:
                            with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                        segment_counter.bad_text_segment_count += 1
                    else:
                        if on_gcs:
                            with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                        segment_counter.over_ctx_len_segment_count += 1

                break
        if len(segments_list) == 0:
            return (None, segment_counter)

        return (segments_list, segment_counter)
    except ValueError as e:
        segment_counter = SegmentCounter()
        segment_counter.failed_transcript_count += 1
        if on_gcs:
            with open(f"{log_dir}/failed_chunking.txt", "a") as f:
                f.write(f"{video_id}\t{e}\n")
        return None, segment_counter
    except Exception as e:
        segment_counter = SegmentCounter()
        segment_counter.failed_transcript_count += 1
        if on_gcs:
            with open(f"{log_dir}/failed_chunking.txt", "a") as f:
                f.write(f"{video_id}\t{e}\n")
        return None, segment_counter


def merge_man_mach_segs(
    transcript: Dict[str, Any],
    shard_log_dir: str,
    in_memory: bool,
) -> Tuple[
    Optional[List[Segment]],
    int,
    int,
    int,
    Optional[SegmentCounter],
    Optional[SegmentCounter],
]:
    """
    Merge manual and machine transcript segments.

    Args:
        transcript: Dictionary containing transcript content and metadata
        shard_log_dir: Directory to save log files for error tracking
        in_memory: If True, keep segments in memory instead of writing to disk

    Returns:
        Tuple containing:
            - List of Segment objects (or None if no valid segments)
            - Boolean indicating if a manual segment failed to be chunked
            - Boolean indicating if a machine transcript failed to be chunked
            - Boolean indicating if a manual transcript might not be good to use
            - SegmentCounter object containing statistics about the manual transcript chunking
            - SegmentCounter object containing statistics about the machine transcript chunking
    """
    result = chunk_transcript_only(
        transcript_data=transcript,
        in_memory=in_memory,
    )

    segments, man_counts = result

    if segments is not None:
        # Convert man_counts to SegmentCounter if it's an int
        if isinstance(man_counts, int):
            man_counts = SegmentCounter()

        shard_mach_log_dir = shard_log_dir + "_mach"
        os.makedirs(shard_mach_log_dir, exist_ok=True)

        # Convert list of lists to list of tuples for proper typing
        man_timestamps = []
        for segment in segments:
            timestamp_parts = segment.text_timestamp.split("_")
            if len(timestamp_parts) >= 2:
                man_timestamps.append((timestamp_parts[0], timestamp_parts[1]))
            else:
                # Handle case where timestamp doesn't have expected format
                man_timestamps.append((timestamp_parts[0], timestamp_parts[0]))

        mach_segments, mach_counts_result = chunk_mach_transcript(
            transcript_data=transcript,
            log_dir=shard_mach_log_dir,
            man_timestamps=man_timestamps,
            in_memory=in_memory,
        )

        # Convert mach_counts to SegmentCounter if it's an int
        if isinstance(mach_counts_result, int):
            mach_counts = SegmentCounter()
        else:
            mach_counts = mach_counts_result

        new_segments = []

        if mach_segments is None:
            for segment in segments:
                segment.add_attr("mach_seg_content", "None")
                segment.add_attr("seg_text", "None")
                segment.add_attr("mach_seg_text", "None")
                segment.add_attr("mach_timestamp", "")
                segment.add_attr("seg_edit_dist", 0.0)
                new_segments.append(segment)
        else:
            mach_segments_deque = deque(mach_segments)
            for segment in segments:
                seg_text = get_seg_text(segment)
                if len(mach_segments_deque) == 0:
                    try:
                        norm_seg_text = normalizer(seg_text).strip()
                    except Exception:
                        norm_seg_text = seg_text
                    segment.add_attr("seg_text", norm_seg_text)
                    segment.add_attr("mach_seg_text", "")
                    segment.add_attr("mach_seg_content", "")
                    segment.add_attr("mach_timestamp", "")
                    if norm_seg_text != "":
                        edit_dist = jiwer.wer(norm_seg_text, "")
                    elif seg_text != "":
                        edit_dist = jiwer.wer(seg_text, "")
                    elif seg_text == "":
                        edit_dist = 0.0
                    segment.add_attr("seg_edit_dist", edit_dist)
                    new_segments.append(segment)
                else:
                    mach_segment = mach_segments_deque.popleft()
                    mach_seg_text = get_mach_seg_text(mach_segment)
                    try:
                        norm_mach_seg_text = normalizer(mach_seg_text).strip()
                    except Exception:
                        norm_mach_seg_text = mach_seg_text
                    try:
                        norm_seg_text = normalizer(seg_text).strip()
                    except Exception:
                        norm_seg_text = seg_text

                    if norm_seg_text != "":
                        edit_dist = jiwer.wer(norm_seg_text, norm_mach_seg_text)
                    elif seg_text == "":
                        if norm_mach_seg_text != "":
                            edit_dist = jiwer.wer(norm_mach_seg_text, seg_text)
                        elif mach_seg_text != "":
                            edit_dist = jiwer.wer(mach_seg_text, seg_text)
                        elif mach_seg_text == "":
                            edit_dist = 0.0
                    elif seg_text != "":
                        edit_dist = jiwer.wer(seg_text, norm_mach_seg_text)
                    segment.add_attr(
                        "seg_text", (norm_seg_text if norm_seg_text != "" else seg_text)
                    )
                    segment.add_attr(
                        "mach_seg_text",
                        (
                            norm_mach_seg_text
                            if norm_mach_seg_text != ""
                            else mach_seg_text
                        ),
                    )
                    segment.add_attr("mach_seg_content", mach_segment.seg_content)
                    segment.add_attr("mach_timestamp", mach_segment.timestamp)
                    segment.add_attr("seg_edit_dist", edit_dist)

                    new_segments.append(segment)

            # if there are remaining mach_segments, the manual transcript might not be good to use so discard
            if len(mach_segments_deque) > 0:
                return None, 0, 0, 1, None, None
        segments = new_segments

        return (
            segments,
            0,
            1 if mach_segments is None else 0,
            0,
            man_counts,
            None if mach_segments is None else mach_counts,
        )
    else:
        # Convert man_counts to SegmentCounter if it's an int
        if isinstance(man_counts, int):
            man_counts = SegmentCounter()
        return None, 1, 0, 0, man_counts, None


def preprocess_jsonl(
    json_file: str,
    shard: str,
    log_dir: str,
    output_dir: str,
    only_subsample: bool = False,
    subsample: bool = False,
    subsample_size: int = 0,
    subsample_seed: int = 42,
    seg_mach: bool = False,
    in_memory: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Preprocess a JSONL file into segments.

    Args:
        json_file: Path to the JSONL file to preprocess
        shard: Identifier for the shard
        log_dir: Directory to save log files for error tracking
        output_dir: Directory to save the preprocessed segments
        only_subsample: If True, only subsample the segments
        subsample: If True, subsample the segments
        subsample_size: Size of the subsample
        subsample_seed: Seed for the subsample
        seg_mach: If True, merge manual and machine transcript segments
        in_memory: If True, keep segments in memory instead of writing to disk

    Returns:
        Dictionary containing statistics about the preprocessed segments,
        or None if only_subsample is True and file already exists
    """
    output_path = f"{output_dir}/shard_seg_{shard}.jsonl.gz"
    stats_path = f"{output_dir}/shard_seg_{shard}_stats.json"
    if os.path.exists(output_path):
        if not only_subsample:
            with open(stats_path, "r") as f:
                stats = json.load(f)
            return stats
        else:
            return None
    else:
        stats = {}

        if not only_subsample:
            shard_log_dir = os.path.join(log_dir, shard)
            os.makedirs(shard_log_dir, exist_ok=True)

            if json_file.endswith(".gz"):
                transcript_data = unarchive_jsonl_gz(json_file)
            else:
                with open(json_file, "r") as f:
                    transcript_data = [json.loads(line.strip()) for line in f]

            if seg_mach == False:
                results = []
                for transcript in transcript_data:
                    result = chunk_transcript_only(transcript, in_memory)
                    results.append(result)

                segments_group = [
                    result[0] for result in results if result[0] is not None
                ]
                man_seg_stats = [
                    result[1]
                    for result in results
                    if isinstance(result[1], SegmentCounter)
                ]
            else:
                results = []
                for transcript in transcript_data:
                    result = merge_man_mach_segs(
                        transcript=transcript,
                        shard_log_dir=shard_log_dir,
                        in_memory=in_memory,
                    )
                    results.append(result)

                segments_group = [
                    result[0] for result in results if result[0] is not None
                ]
                count_no_seg = [result[1] for result in results]
                count_failed_mach_seg = [result[2] for result in results]
                count_bad_man_transcripts = [result[3] for result in results]
                man_seg_stats = [
                    result[4] for result in results if result[4] is not None
                ]
                mach_seg_stats = [
                    result[5] for result in results if result[5] is not None
                ]

            segments_list = list(chain(*segments_group))
            seg_count = len(segments_list)

            if seg_mach == True:
                stats["no_segment_video_id_count"] = sum(count_no_seg)
                stats["failed_mach_segment_video_id_count"] = sum(count_failed_mach_seg)
                stats["bad_man_transcript_video_id_count"] = sum(
                    count_bad_man_transcripts
                )

            stats["pre_segment_video_id_count"] = len(transcript_data)
            stats["post_segment_video_id_count"] = len(segments_group)
            stats["total_segment_count"] = seg_count

            stats["man_seg_stats"] = sum_counters(man_seg_stats).to_dict()
            if seg_mach == True:
                stats["mach_seg_stats"] = sum_counters(mach_seg_stats).to_dict()
            else:
                stats["mach_seg_stats"] = SegmentCounter().to_dict()
        else:
            with gzip.open(json_file, "rt", encoding="utf-8") as f:
                segments_list = [json.loads(line.strip()) for line in f]
            seg_count = len(segments_list)
            stats["total_segment_count"] = seg_count

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

            stats["total_subsampled_segment_count"] = subsampled_count

            with gzip.open(output_path, "wt", encoding="utf-8") as f:
                for segment in segments_list:
                    if hasattr(segment, "to_dict"):
                        f.write(json.dumps(segment.to_dict()) + "\n")
                    else:
                        f.write(json.dumps(segment) + "\n")
        else:
            with gzip.open(output_path, "wt", encoding="utf-8") as f:
                for segment in segments_list:
                    if hasattr(segment, "to_dict"):
                        f.write(json.dumps(segment.to_dict()) + "\n")
                    else:
                        f.write(json.dumps(segment) + "\n")

        if not only_subsample:
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)

        return stats


def parallel_preprocess_jsonl(
    args: Tuple[str, str, str, str, bool, bool, int, int, bool, bool]
) -> Optional[Dict[str, Any]]:
    """
    Parallel wrapper for preprocess_jsonl function.

    Args:
        args: Tuple containing all arguments for preprocess_jsonl

    Returns:
        Result from preprocess_jsonl function
    """
    return preprocess_jsonl(*args)


def preprocess_jsonls(
    input_dir: str,
    output_dir: str,
    log_dir: str,
    only_subsample: bool = False,
    subsample: bool = False,
    subsample_size: int = 0,
    subsample_seed: int = 42,
    seg_mach: bool = False,
    in_memory: bool = True,
) -> None:
    """
    Preprocess multiple JSONL files in parallel.

    Args:
        input_dir: Directory containing JSONL files to preprocess
        output_dir: Directory to save preprocessed segments
        log_dir: Directory to save log files for error tracking
        only_subsample: If True, only subsample existing segments
        subsample: If True, subsample the segments
        subsample_size: Size of the subsample
        subsample_seed: Seed for the subsample
        seg_mach: If True, merge manual and machine transcript segments
        in_memory: If True, keep segments in memory instead of writing to disk
    """
    print(
        f"{only_subsample=}, {subsample=}, {subsample_size=}, {subsample_seed=}, {in_memory=}"
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    shard_jsonls = glob.glob(f"{input_dir}/*.jsonl.gz")
    get_shard = lambda shard_jsonl: shard_jsonl.split("_")[-1].split(".")[0]

    shards = [get_shard(shard_jsonl) for shard_jsonl in shard_jsonls]
    print(f"{len(shards)} shards found")
    print(f"{shards[:5]=}")

    with multiprocessing.Pool() as pool:
        stats = list(
            tqdm(
                pool.imap_unordered(
                    parallel_preprocess_jsonl,
                    zip(
                        shard_jsonls,
                        shards,
                        repeat(log_dir),
                        repeat(output_dir),
                        repeat(only_subsample),
                        repeat(subsample),
                        repeat(subsample_size),
                        repeat(subsample_seed),
                        repeat(seg_mach),
                        repeat(in_memory),
                    ),
                ),
                total=len(shard_jsonls),
            )
        )

    stats = [stat for stat in stats if stat is not None]

    with open(f"{output_dir}/segment_stats.log", "w") as f:
        if stats:
            stat_keys = stats[0].keys()
            for key in stat_keys:
                if key != "man_seg_stats" and key != "mach_seg_stats":
                    f.write(f"{key}: {sum([stat[key] for stat in stats])}\n")
                    if "video_id" in key:
                        f.write(
                            f"dur_by_{key}: {(sum([stat[key] for stat in stats]) * 30) / (60 * 60)} hours\n"
                        )
                else:
                    # Handle nested dictionaries for segment stats
                    if stats[0][key]:
                        nested_keys = stats[0][key].keys()
                        f.write(f"{key}:\n")
                        for nested_key in nested_keys:
                            total_value = sum(
                                [stat[key][nested_key] for stat in stats if stat[key]]
                            )
                            f.write(f"  {nested_key}: {total_value}\n")
                    else:
                        f.write(f"{key}: {{}}\n")


if __name__ == "__main__":
    Fire(
        {
            "preprocess_jsonls": preprocess_jsonls,
            "preprocess_jsonl": preprocess_jsonl,
            "merge_man_mach_segs": merge_man_mach_segs,
            "chunk_mach_transcript": chunk_mach_transcript,
            "chunk_transcript_only": chunk_transcript_only,
            "chunk_data": chunk_data,
            "chunk_local": chunk_local,
            "chunk_gcs": chunk_gcs,
            "unarchive_jsonl_gz": unarchive_jsonl_gz,
        }
    )
