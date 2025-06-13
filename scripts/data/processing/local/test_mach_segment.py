# %%
import sys
import os

sys.path.append("/Users/huongn/Desktop/open_whisper/scripts/data/processing/local")
import segment_jsonl_utils as utils
import json
import glob
from typing import List, Dict, Tuple, Optional
import numpy as np
import pysrt
import webvtt
from collections import deque
from whisper.normalizers import EnglishTextNormalizer
import jiwer
import Levenshtein
from open_whisper.utils import TranscriptReader

SEGMENT_COUNT_THRESHOLD = 120


# %%
def chunk_transcript(
    transcript_data: Dict,
    keep_tokens: bool = False,
    dolma_format: bool = False,
    mach_transcript: bool = False,
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
        # if transcript_file.startswith("/weka"):
        #     video_id = transcript_file.split("/")[5]
        # else:
        #     video_id = transcript_file.split("/")[1]
        video_id = "test"

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
            print(f"Empty transcript for {video_id}")
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
                    print(f"Transcript line > 30s for {video_id}")

                    a += 1
                    b += 1

                    if a == b == len(transcript):
                        if segment_count == 0:
                            print(f"Transcript line > 30s for {video_id}")
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
                                "timestamp": timestamp,
                                "id": video_id,
                                "seg_id": f"{video_id}_{segment_count}",
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
                        print(f"Faulty transcript for {video_id}")
                    elif res is None:
                        print(f"Faulty transcript for {video_id}")

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
                                "timestamp": timestamp,
                                "id": video_id,
                                "seg_id": f"{video_id}_{segment_count}",
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
                        print(f"Faulty transcript for {video_id}")
                    elif res is None:
                        print(f"Faulty transcript for {video_id}")

                break
        if len(segments_list) == 0:
            return None
        return segments_list
    except ValueError as e:
        print(f"ValueError: {e}")
        return None
    except Exception as e:
        print(f"Exception: {e}")
        return None


# %%
def chunk_mach_transcript(
    transcript_data: Dict,
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
        # if transcript_file.startswith("/weka"):
        #     video_id = transcript_file.split("/")[5]
        # else:
        #     video_id = transcript_file.split("/")[1]
        video_id = "test"

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
            print(f"Empty transcript for {video_id}")
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
        
        b = a        
        while (
            a < len(transcript) + 1
            and segment_count < SEGMENT_COUNT_THRESHOLD
            and man_seg_idx < len(man_timestamps)
        ):
            init_diff = utils.calculate_difference(timestamps[a][0], timestamps[b][1])

            # if init_diff < 30000 or utils.convert_to_milliseconds(
            # if utils.convert_to_milliseconds(
            #     timestamps[b][1]
            # ) <= utils.convert_to_milliseconds(man_timestamps[man_seg_idx][1]):
            man_mach_diff = np.absolute(
                utils.convert_to_milliseconds(man_timestamps[man_seg_idx][1])
                - utils.convert_to_milliseconds(timestamps[b][1])
            )
            if man_mach_diff <= max_man_mach_diff:
                diff = init_diff
                max_man_mach_diff = man_mach_diff
                b += 1
            # elif init_diff >= 30000 or utils.convert_to_milliseconds(
            # elif utils.convert_to_milliseconds(
            #     timestamps[b][1]
            # ) > utils.convert_to_milliseconds(man_timestamps[man_seg_idx][1]):
            elif man_mach_diff > max_man_mach_diff:
                # edge case (when transcript line is > 30s)
                if b == a:
                    print(f"Transcript line > 30s for {video_id}")

                    a += 1
                    b += 1

                    if a == b == len(transcript):
                        if segment_count == 0:
                            print(f"Transcript line > 30s for {video_id}")
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
                            "timestamp": timestamp,
                            "id": video_id,
                            # "seg_id": f"{video_id}_{segment_count}",
                            "audio_file": t_output_file.replace(
                                f".{transcript_ext}", ".npy"
                            ).replace("ow_full", "ow_seg"),
                        }
                        segments_list.append(segment)
                        segment_count += 1
                        man_seg_idx += 1
                else:
                    if type(res) is not List or res is not None:
                        print(f"Faulty transcript for {video_id}")
                    elif res is None:
                        print(f"Faulty transcript for {video_id}")

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
                            "timestamp": timestamp,
                            "id": video_id,
                            # "seg_id": f"{video_id}_{segment_count}",
                            "audio_file": t_output_file.replace(
                                f".{transcript_ext}", ".npy"
                            ).replace("ow_full", "ow_seg"),
                        }
                        segments_list.append(segment)
                        segment_count += 1
                        man_seg_idx += 1
                else:
                    if type(res) is not List or res is not None:
                        print(f"Faulty transcript for {video_id}")
                    elif res is None:
                        print(f"Faulty transcript for {video_id}")

                break
        if len(segments_list) == 0:
            return None

        return segments_list
    except ValueError as e:
        print(f"ValueError: {e}")
        return None
    except Exception as e:
        print(f"Exception: {e}")
        return None


# %%
content = """1
00:00:06,160 --> 00:00:07,880
Why we became furless

2
00:00:08,060 --> 00:00:09,880
has long fascinated scientist,

3
00:00:09,880 --> 00:00:12,400
who have proposed a number of explanations,

4
00:00:12,540 --> 00:00:14,920
but the majority of researchers today postulate

5
00:00:14,920 --> 00:00:18,700
that reduced body hair had to do with Thermoregulation-

6
00:00:19,100 --> 00:00:21,120
specifically, with keeping cool.

7
00:00:22,640 --> 00:00:26,180
The core argument is: During the evolutionary phase

8
00:00:26,180 --> 00:00:28,080
after our ancestors became bipeds,

9
00:00:28,080 --> 00:00:30,880
they were regularly walking or running in open,

10
00:00:31,060 --> 00:00:33,180
drier habitats.

11
00:00:33,240 --> 00:00:35,560
Imagine a patchy woodland or savannah,

12
00:00:35,740 --> 00:00:39,000
rather than a dense, shady rainforest.

13
00:00:40,080 --> 00:00:42,760
In such a context, overheating was a serious risk.

14
00:00:42,880 --> 00:00:45,480
Reduced body hair and increased sweat  glands

15
00:00:45,880 --> 00:00:48,080
were favored because this allowed for more

16
00:00:48,080 --> 00:00:51,960
effective evaporative cooling via perspiration.

17
00:00:53,560 --> 00:00:55,920
In comparison, most furry mammals

18
00:00:56,640 --> 00:00:57,140
pant

19
00:00:57,560 --> 00:00:59,720
to regulate their body temperature.

20
00:00:59,880 --> 00:01:03,100
Other animals like lizards, amphibians and insects

21
00:01:03,100 --> 00:01:06,180
have other behaviors that help keep them cool.

22
00:01:06,780 --> 00:01:09,760
Humans, however, are in a category of their own.

23
00:01:10,060 --> 00:01:12,760
We are the only mammals that relies on secreating

24
00:01:12,760 --> 00:01:16,000
water onto the surface of our skin to stay cool.

25
00:01:16,960 --> 00:01:18,320
But why?

26
00:01:18,320 --> 00:01:19,600
While sweating might lead

27
00:01:19,600 --> 00:01:22,520
to awkward encounters on a hot day, scientist believe

28
00:01:22,520 --> 00:01:25,020
that it also gave us an evolutionary advantage.

29
00:01:25,020 --> 00:01:27,980
Our ability to sweat let us run longer distances

30
00:01:28,080 --> 00:01:31,180
at faster speeds than other animals

31
00:01:31,180 --> 00:01:34,400
This meant humans could hunt game during  the  hottest

32
00:01:34,540 --> 00:01:37,280
parts of the day, when other predators were forced

33
00:01:37,500 --> 00:01:38,080
to rest

34
00:01:39,820 --> 00:01:43,360
This elevated activity levels came at a price:

35
00:01:43,360 --> 00:01:45,980
a greatly increased risk of overheating.

36
00:01:46,300 --> 00:01:48,640
The increase in walking  and running,

37
00:01:48,640 --> 00:01:52,140
during which muscle activity builds up heat internally,

38
00:01:52,140 --> 00:01:55,180
would have required  that hominids both

39
00:01:55,180 --> 00:01:57,980
enhance their sweating ability and lose

40
00:01:57,980 --> 00:02:01,060
their body hair to avoid overheating.

41
00:02:01,400 --> 00:02:04,225
And lastly, having hair on our palms

42
00:02:04,225 --> 00:02:06,620
and wrist would make knapping stone tools

43
00:02:06,820 --> 00:02:10,020
or operating machinery rather difficult, and so humans

44
00:02:10,020 --> 00:02:12,775
ancestors who lost this hair may have had an advantage.

45
00:02:13,680 --> 00:02:15,535
So who knew that losing hair and becoming

46
00:02:15,540 --> 00:02:18,060
very sweaty would be a positve rather than

47
00:02:18,200 --> 00:02:21,840
a negative and allowed us to hunt , run long distances

48
00:02:21,840 --> 00:02:24,360
and adapt to earth as it changed.


"""
mach_content = """WEBVTT
Kind: captions
Language: en

00:00:05.569 --> 00:00:08.030 align:start position:0%
while<00:00:06.569><c> we</c><00:00:06.750><c> became</c><00:00:07.170><c> furless</c><00:00:07.680><c> has</c><00:00:07.859><c> long</c>

00:00:08.030 --> 00:00:08.040 align:start position:0%
while we became furless has long


00:00:08.040 --> 00:00:10.549 align:start position:0%
while we became furless has long
fascinated<00:00:08.280><c> scientists</c><00:00:09.269><c> who</c><00:00:10.050><c> have</c><00:00:10.200><c> proposed</c>

00:00:10.549 --> 00:00:10.559 align:start position:0%
fascinated scientists who have proposed


00:00:10.559 --> 00:00:12.440 align:start position:0%
fascinated scientists who have proposed
a<00:00:10.650><c> number</c><00:00:11.010><c> of</c><00:00:11.190><c> explanations</c><00:00:11.490><c> but</c><00:00:12.300><c> the</c>

00:00:12.440 --> 00:00:12.450 align:start position:0%
a number of explanations but the


00:00:12.450 --> 00:00:14.690 align:start position:0%
a number of explanations but the
majority<00:00:12.840><c> of</c><00:00:12.870><c> researchers</c><00:00:13.440><c> today</c><00:00:13.700><c> postulate</c>

00:00:14.690 --> 00:00:14.700 align:start position:0%
majority of researchers today postulate


00:00:14.700 --> 00:00:17.150 align:start position:0%
majority of researchers today postulate
that<00:00:14.880><c> reduced</c><00:00:15.299><c> body</c><00:00:15.570><c> hair</c><00:00:15.990><c> had</c><00:00:16.440><c> to</c><00:00:16.590><c> do</c><00:00:16.770><c> with</c>

00:00:17.150 --> 00:00:17.160 align:start position:0%
that reduced body hair had to do with


00:00:17.160 --> 00:00:20.240 align:start position:0%
that reduced body hair had to do with
thermo<00:00:17.730><c> regulation</c><00:00:18.619><c> specifically</c><00:00:19.619><c> with</c>

00:00:20.240 --> 00:00:20.250 align:start position:0%
thermo regulation specifically with


00:00:20.250 --> 00:00:24.500 align:start position:0%
thermo regulation specifically with
keeping<00:00:20.640><c> cool</c><00:00:21.650><c> the</c><00:00:22.650><c> core</c><00:00:22.920><c> argument</c><00:00:23.460><c> is</c><00:00:23.609><c> during</c>

00:00:24.500 --> 00:00:24.510 align:start position:0%
keeping cool the core argument is during


00:00:24.510 --> 00:00:26.359 align:start position:0%
keeping cool the core argument is during
the<00:00:24.660><c> evolutionary</c><00:00:25.230><c> phase</c><00:00:25.619><c> after</c><00:00:25.949><c> our</c>

00:00:26.359 --> 00:00:26.369 align:start position:0%
the evolutionary phase after our


00:00:26.369 --> 00:00:28.640 align:start position:0%
the evolutionary phase after our
ancestors<00:00:26.939><c> became</c><00:00:27.300><c> bipeds</c><00:00:27.869><c> they</c><00:00:28.590><c> were</c>

00:00:28.640 --> 00:00:28.650 align:start position:0%
ancestors became bipeds they were


00:00:28.650 --> 00:00:31.099 align:start position:0%
ancestors became bipeds they were
regularly<00:00:29.099><c> walking</c><00:00:29.849><c> or</c><00:00:30.000><c> running</c><00:00:30.300><c> and</c><00:00:30.630><c> open</c>

00:00:31.099 --> 00:00:31.109 align:start position:0%
regularly walking or running and open


00:00:31.109 --> 00:00:32.720 align:start position:0%
regularly walking or running and open
drier<00:00:31.740><c> habitats</c>

00:00:32.720 --> 00:00:32.730 align:start position:0%
drier habitats


00:00:32.730 --> 00:00:35.270 align:start position:0%
drier habitats
imagine<00:00:33.390><c> a</c><00:00:33.480><c> patchy</c><00:00:34.020><c> woolen</c><00:00:34.559><c> or</c><00:00:34.770><c> savanna</c>

00:00:35.270 --> 00:00:35.280 align:start position:0%
imagine a patchy woolen or savanna


00:00:35.280 --> 00:00:39.619 align:start position:0%
imagine a patchy woolen or savanna
rather<00:00:36.239><c> than</c><00:00:36.450><c> a</c><00:00:36.570><c> dense</c><00:00:36.809><c> shady</c><00:00:37.530><c> rainforest</c><00:00:38.629><c> in</c>

00:00:39.619 --> 00:00:39.629 align:start position:0%
rather than a dense shady rainforest in


00:00:39.629 --> 00:00:41.750 align:start position:0%
rather than a dense shady rainforest in
such<00:00:39.840><c> a</c><00:00:39.870><c> context</c><00:00:40.469><c> overheating</c><00:00:41.160><c> was</c><00:00:41.309><c> a</c><00:00:41.340><c> serious</c>

00:00:41.750 --> 00:00:41.760 align:start position:0%
such a context overheating was a serious


00:00:41.760 --> 00:00:44.450 align:start position:0%
such a context overheating was a serious
risk<00:00:42.260><c> reduced</c><00:00:43.260><c> body</c><00:00:43.440><c> hair</c><00:00:43.829><c> and</c><00:00:43.860><c> increased</c>

00:00:44.450 --> 00:00:44.460 align:start position:0%
risk reduced body hair and increased


00:00:44.460 --> 00:00:46.610 align:start position:0%
risk reduced body hair and increased
sweat<00:00:44.820><c> glands</c><00:00:45.120><c> were</c><00:00:45.600><c> favored</c><00:00:46.020><c> because</c><00:00:46.320><c> it</c>

00:00:46.610 --> 00:00:46.620 align:start position:0%
sweat glands were favored because it


00:00:46.620 --> 00:00:49.549 align:start position:0%
sweat glands were favored because it
allowed<00:00:46.950><c> for</c><00:00:47.610><c> more</c><00:00:47.820><c> effective</c><00:00:48.559><c> evaporative</c>

00:00:49.549 --> 00:00:49.559 align:start position:0%
allowed for more effective evaporative


00:00:49.559 --> 00:00:54.290 align:start position:0%
allowed for more effective evaporative
cooling<00:00:49.890><c> by</c><00:00:50.520><c> perspiration</c><00:00:52.520><c> in</c><00:00:53.520><c> comparison</c>

00:00:54.290 --> 00:00:54.300 align:start position:0%
cooling by perspiration in comparison


00:00:54.300 --> 00:00:58.250 align:start position:0%
cooling by perspiration in comparison
most<00:00:54.840><c> furry</c><00:00:55.230><c> mammals</c><00:00:55.739><c> had</c><00:00:56.699><c> to</c><00:00:57.600><c> regulate</c><00:00:57.960><c> their</c>

00:00:58.250 --> 00:00:58.260 align:start position:0%
most furry mammals had to regulate their


00:00:58.260 --> 00:01:00.439 align:start position:0%
most furry mammals had to regulate their
body<00:00:58.410><c> temperature</c><00:00:58.699><c> other</c><00:00:59.699><c> animals</c><00:01:00.270><c> like</c>

00:01:00.439 --> 00:01:00.449 align:start position:0%
body temperature other animals like


00:01:00.449 --> 00:01:03.200 align:start position:0%
body temperature other animals like
lizards<00:01:01.100><c> amphibians</c><00:01:02.100><c> and</c><00:01:02.340><c> insects</c><00:01:02.820><c> have</c>

00:01:03.200 --> 00:01:03.210 align:start position:0%
lizards amphibians and insects have


00:01:03.210 --> 00:01:06.190 align:start position:0%
lizards amphibians and insects have
other<00:01:03.510><c> behaviors</c><00:01:04.110><c> that</c><00:01:04.379><c> help</c><00:01:04.680><c> keep</c><00:01:05.339><c> them</c><00:01:05.580><c> cool</c>

00:01:06.190 --> 00:01:06.200 align:start position:0%
other behaviors that help keep them cool


00:01:06.200 --> 00:01:08.840 align:start position:0%
other behaviors that help keep them cool
humans<00:01:07.200><c> however</c><00:01:07.320><c> are</c><00:01:07.799><c> in</c><00:01:08.070><c> a</c><00:01:08.159><c> category</c><00:01:08.310><c> of</c>

00:01:08.840 --> 00:01:08.850 align:start position:0%
humans however are in a category of


00:01:08.850 --> 00:01:11.149 align:start position:0%
humans however are in a category of
their<00:01:09.180><c> own</c><00:01:09.390><c> we're</c><00:01:10.229><c> the</c><00:01:10.380><c> only</c><00:01:10.409><c> mammal</c><00:01:11.040><c> that</c>

00:01:11.149 --> 00:01:11.159 align:start position:0%
their own we're the only mammal that


00:01:11.159 --> 00:01:13.399 align:start position:0%
their own we're the only mammal that
relies<00:01:11.580><c> on</c><00:01:11.610><c> secreting</c><00:01:12.479><c> water</c><00:01:12.869><c> onto</c><00:01:13.170><c> the</c>

00:01:13.399 --> 00:01:13.409 align:start position:0%
relies on secreting water onto the


00:01:13.409 --> 00:01:17.149 align:start position:0%
relies on secreting water onto the
surface<00:01:13.890><c> of</c><00:01:14.040><c> our</c><00:01:14.430><c> skin</c><00:01:14.729><c> to</c><00:01:15.600><c> stay</c><00:01:15.840><c> cool</c><00:01:16.409><c> but</c>

00:01:17.149 --> 00:01:17.159 align:start position:0%
surface of our skin to stay cool but


00:01:17.159 --> 00:01:19.219 align:start position:0%
surface of our skin to stay cool but
wife<00:01:17.400><c> while</c><00:01:18.270><c> sweating</c><00:01:18.689><c> might</c><00:01:18.960><c> lead</c><00:01:19.200><c> to</c>

00:01:19.219 --> 00:01:19.229 align:start position:0%
wife while sweating might lead to


00:01:19.229 --> 00:01:21.050 align:start position:0%
wife while sweating might lead to
awkward<00:01:19.590><c> encounters</c><00:01:20.310><c> on</c><00:01:20.460><c> a</c><00:01:20.490><c> hot</c><00:01:20.729><c> day</c>

00:01:21.050 --> 00:01:21.060 align:start position:0%
awkward encounters on a hot day


00:01:21.060 --> 00:01:23.300 align:start position:0%
awkward encounters on a hot day
scientists<00:01:21.960><c> believe</c><00:01:22.080><c> that</c><00:01:22.290><c> it</c><00:01:22.619><c> also</c><00:01:22.830><c> gave</c><00:01:23.159><c> us</c>

00:01:23.300 --> 00:01:23.310 align:start position:0%
scientists believe that it also gave us


00:01:23.310 --> 00:01:25.700 align:start position:0%
scientists believe that it also gave us
an<00:01:23.460><c> evolutionary</c><00:01:23.780><c> advantage</c><00:01:24.780><c> our</c><00:01:25.229><c> ability</c><00:01:25.680><c> to</c>

00:01:25.700 --> 00:01:25.710 align:start position:0%
an evolutionary advantage our ability to


00:01:25.710 --> 00:01:26.330 align:start position:0%
an evolutionary advantage our ability to
sweat

00:01:26.330 --> 00:01:26.340 align:start position:0%
sweat


00:01:26.340 --> 00:01:28.670 align:start position:0%
sweat
let<00:01:26.520><c> us</c><00:01:26.640><c> run</c><00:01:26.820><c> longer</c><00:01:27.270><c> distances</c><00:01:27.450><c> at</c><00:01:28.140><c> a</c><00:01:28.170><c> faster</c>

00:01:28.670 --> 00:01:28.680 align:start position:0%
let us run longer distances at a faster


00:01:28.680 --> 00:01:30.590 align:start position:0%
let us run longer distances at a faster
speed<00:01:28.710><c> than</c><00:01:29.670><c> other</c><00:01:29.909><c> animals</c>

00:01:30.590 --> 00:01:30.600 align:start position:0%
speed than other animals


00:01:30.600 --> 00:01:33.740 align:start position:0%
speed than other animals
this<00:01:31.350><c> meant</c><00:01:31.680><c> humans</c><00:01:32.220><c> could</c><00:01:32.430><c> hunt</c><00:01:32.790><c> game</c><00:01:33.180><c> during</c>

00:01:33.740 --> 00:01:33.750 align:start position:0%
this meant humans could hunt game during


00:01:33.750 --> 00:01:35.929 align:start position:0%
this meant humans could hunt game during
the<00:01:33.810><c> hottest</c><00:01:34.079><c> parts</c><00:01:34.530><c> of</c><00:01:34.650><c> the</c><00:01:34.799><c> day</c><00:01:34.860><c> when</c><00:01:35.670><c> other</c>

00:01:35.929 --> 00:01:35.939 align:start position:0%
the hottest parts of the day when other


00:01:35.939 --> 00:01:39.649 align:start position:0%
the hottest parts of the day when other
predators<00:01:36.540><c> were</c><00:01:36.810><c> forced</c><00:01:37.200><c> to</c><00:01:37.320><c> rest</c><00:01:38.659><c> this</c>

00:01:39.649 --> 00:01:39.659 align:start position:0%
predators were forced to rest this


00:01:39.659 --> 00:01:42.140 align:start position:0%
predators were forced to rest this
elevated<00:01:40.200><c> activity</c><00:01:40.710><c> levels</c><00:01:41.310><c> came</c><00:01:41.610><c> at</c><00:01:41.850><c> a</c><00:01:41.909><c> price</c>

00:01:42.140 --> 00:01:42.150 align:start position:0%
elevated activity levels came at a price


00:01:42.150 --> 00:01:45.609 align:start position:0%
elevated activity levels came at a price
a<00:01:42.600><c> greatly</c><00:01:43.409><c> increased</c><00:01:44.189><c> risk</c><00:01:44.490><c> of</c><00:01:44.759><c> overheating</c>

00:01:45.609 --> 00:01:45.619 align:start position:0%
a greatly increased risk of overheating


00:01:45.619 --> 00:01:48.170 align:start position:0%
a greatly increased risk of overheating
the<00:01:46.619><c> increase</c><00:01:47.009><c> in</c><00:01:47.159><c> walking</c><00:01:47.700><c> and</c><00:01:47.880><c> running</c>

00:01:48.170 --> 00:01:48.180 align:start position:0%
the increase in walking and running


00:01:48.180 --> 00:01:50.569 align:start position:0%
the increase in walking and running
during<00:01:48.899><c> which</c><00:01:49.200><c> muscle</c><00:01:49.590><c> activity</c><00:01:49.829><c> builds</c><00:01:50.399><c> up</c>

00:01:50.569 --> 00:01:50.579 align:start position:0%
during which muscle activity builds up


00:01:50.579 --> 00:01:53.480 align:start position:0%
during which muscle activity builds up
heat<00:01:50.909><c> internally</c><00:01:51.720><c> would</c><00:01:52.350><c> have</c><00:01:52.560><c> required</c><00:01:52.920><c> that</c>

00:01:53.480 --> 00:01:53.490 align:start position:0%
heat internally would have required that


00:01:53.490 --> 00:01:56.539 align:start position:0%
heat internally would have required that
hominids<00:01:54.090><c> both</c><00:01:54.600><c> enhance</c><00:01:55.320><c> their</c><00:01:55.920><c> sweating</c>

00:01:56.539 --> 00:01:56.549 align:start position:0%
hominids both enhance their sweating


00:01:56.549 --> 00:01:59.420 align:start position:0%
hominids both enhance their sweating
ability<00:01:57.030><c> and</c><00:01:57.270><c> lose</c><00:01:57.719><c> their</c><00:01:58.140><c> body</c><00:01:58.350><c> hair</c><00:01:58.710><c> to</c>

00:01:59.420 --> 00:01:59.430 align:start position:0%
ability and lose their body hair to


00:01:59.430 --> 00:02:02.959 align:start position:0%
ability and lose their body hair to
avoid<00:01:59.670><c> overheating</c><00:02:00.270><c> and</c><00:02:01.159><c> lastly</c><00:02:02.159><c> having</c><00:02:02.610><c> hair</c>

00:02:02.959 --> 00:02:02.969 align:start position:0%
avoid overheating and lastly having hair


00:02:02.969 --> 00:02:04.940 align:start position:0%
avoid overheating and lastly having hair
on<00:02:03.119><c> our</c><00:02:03.270><c> palms</c><00:02:03.570><c> and</c><00:02:04.079><c> wrists</c><00:02:04.409><c> would</c><00:02:04.710><c> make</c>

00:02:04.940 --> 00:02:04.950 align:start position:0%
on our palms and wrists would make


00:02:04.950 --> 00:02:06.950 align:start position:0%
on our palms and wrists would make
napping<00:02:05.549><c> stone</c><00:02:06.000><c> tools</c><00:02:06.270><c> or</c><00:02:06.509><c> operating</c>

00:02:06.950 --> 00:02:06.960 align:start position:0%
napping stone tools or operating


00:02:06.960 --> 00:02:09.859 align:start position:0%
napping stone tools or operating
machinery<00:02:07.320><c> rather</c><00:02:08.220><c> difficult</c><00:02:08.670><c> and</c><00:02:09.119><c> so</c><00:02:09.509><c> human</c>

00:02:09.859 --> 00:02:09.869 align:start position:0%
machinery rather difficult and so human


00:02:09.869 --> 00:02:12.500 align:start position:0%
machinery rather difficult and so human
ancestors<00:02:10.379><c> who</c><00:02:10.649><c> lost</c><00:02:10.920><c> his</c><00:02:11.280><c> hair</c><00:02:11.610><c> may</c><00:02:12.090><c> have</c><00:02:12.300><c> had</c>

00:02:12.500 --> 00:02:12.510 align:start position:0%
ancestors who lost his hair may have had


00:02:12.510 --> 00:02:13.290 align:start position:0%
ancestors who lost his hair may have had
an<00:02:12.750><c> advanced</c>

00:02:13.290 --> 00:02:13.300 align:start position:0%
an advanced


00:02:13.300 --> 00:02:14.970 align:start position:0%
an advanced
so<00:02:13.660><c> who</c><00:02:13.810><c> knew</c><00:02:14.020><c> that</c><00:02:14.080><c> loosing</c><00:02:14.620><c> hair</c><00:02:14.830><c> and</c>

00:02:14.970 --> 00:02:14.980 align:start position:0%
so who knew that loosing hair and


00:02:14.980 --> 00:02:16.860 align:start position:0%
so who knew that loosing hair and
becoming<00:02:15.310><c> very</c><00:02:15.580><c> sweaty</c><00:02:16.000><c> would</c><00:02:16.420><c> be</c><00:02:16.570><c> a</c><00:02:16.600><c> positive</c>

00:02:16.860 --> 00:02:16.870 align:start position:0%
becoming very sweaty would be a positive


00:02:16.870 --> 00:02:19.440 align:start position:0%
becoming very sweaty would be a positive
rather<00:02:17.440><c> than</c><00:02:17.590><c> a</c><00:02:17.800><c> negative</c><00:02:17.830><c> and</c><00:02:18.820><c> allowed</c><00:02:19.240><c> us</c><00:02:19.420><c> to</c>

00:02:19.440 --> 00:02:19.450 align:start position:0%
rather than a negative and allowed us to


00:02:19.450 --> 00:02:22.710 align:start position:0%
rather than a negative and allowed us to
hunt<00:02:19.840><c> run</c><00:02:20.650><c> long</c><00:02:21.040><c> distances</c><00:02:21.070><c> and</c><00:02:21.940><c> adapt</c><00:02:22.600><c> to</c>

00:02:22.710 --> 00:02:22.720 align:start position:0%
hunt run long distances and adapt to


00:02:22.720 --> 00:02:25.980 align:start position:0%
hunt run long distances and adapt to
earth<00:02:22.990><c> as</c><00:02:23.320><c> it</c><00:02:23.590><c> changed</c>

"""
# %%
# transcript_data = {
#     "content": content,
#     "mach_content": mach_content,
#     "subtitle_file": "test.srt",
# }
# # %%
# segments = chunk_transcript(transcript_data=transcript_data, mach_transcript=False)
# # %%
# mach_segments = chunk_transcript(transcript_data=transcript_data, mach_transcript=True)
# # %%
# normalizer = EnglishTextNormalizer()
# reader = TranscriptReader(transcript_string=segments[0]["seg_content"], ext="srt")
# t_dict, *_ = reader.read()
# segment_text = reader.extract_text(t_dict)
# content = webvtt.from_string(mach_segments[0]["seg_content"])
# modified_content = []
# if content[0].text == content[1].text:
#     modified_content.append(content[0])
#     start = 2
# else:
#     start = 1
# print(f"{start=}")
# for i in range(start, len(content)):
#     caption = content[i]
#     if "\n" not in caption.text:
#         modified_content.append(caption)
# mach_segment_text = " ".join([caption.text for caption in modified_content])
# print(f"{segment_text=}")
# print(f"{mach_segment_text=}")
# # %%
# jiwer.wer(normalizer(segment_text), normalizer(mach_segment_text))
# # %%
# reader = TranscriptReader(transcript_string=segments[0]["seg_content"], ext="srt")
# t_dict, *_ = reader.read()
# segment_text = reader.extract_text(t_dict)
# content = webvtt.from_string(mach_segments[5]["seg_content"])
# modified_content = []
# if content[0].text == content[1].text:
#     modified_content.append(content[0])
#     start = 2
# else:
#     start = 1
# print(f"{start=}")
# for i in range(start, len(content)):
#     caption = content[i]
#     if "\n" not in caption.text:
#         modified_content.append(caption)
# mach_segment_text = " ".join([caption.text for caption in modified_content])
# print(f"{segment_text=}")
# print(f"{mach_segment_text=}")
# # %%
# jiwer.wer(normalizer(segment_text), normalizer(mach_segment_text))

# %%
normalizer = EnglishTextNormalizer()


def get_seg_text(segment):
    reader = TranscriptReader(transcript_string=segment["seg_content"], ext="srt")
    t_dict, *_ = reader.read()
    segment_text = reader.extract_text(t_dict)
    return segment_text


def get_mach_seg_text(mach_segment):
    content = webvtt.from_string(mach_segment["seg_content"])
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


# %%
# from collections import deque

# mach_segments = deque(mach_segments)
# for segment in segments:
#     seg_text = get_seg_text(segment)
#     if seg_text != "":
#         mach_segment = mach_segments.popleft()
#         mach_seg_text = get_mach_seg_text(mach_segment)
#         print(f"{seg_text=}")
#         print(f"{mach_seg_text=}")
#         print(f"{jiwer.wer(normalizer(seg_text), normalizer(mach_seg_text))=}")
# # %%
# from collections import deque

# mach_segments = deque(mach_segments)
# for segment in segments:
#     seg_text = get_seg_text(segment)
#     if seg_text != "":
#         mach_segment = mach_segments.popleft()
#         mach_seg_text = get_mach_seg_text(mach_segment)
#         wer = jiwer.wer(normalizer(seg_text), normalizer(mach_seg_text))
#         if wer < 0.3:
#             segment["mach_seg_content"] = mach_seg_text
#             segment["mach_timestamps"] = mach_segment["timestamp"]
#             segment["mach_seg_id"] = mach_segment["seg_id"]
#             segment["wer"] = wer
#         else:
#             mach_segments.appendleft(mach_segment)
#     else:
#         segment["mach_seg_content"] = ""
#         segment["mach_timestamps"] = ""
#         segment["mach_seg_id"] = ""
#         segment["wer"] = -1.0
# # %%
# segments
# # %%
# for segment in segments:
#     if "mach_seg_content" not in segments:
#         segment["mach_seg_content"] = "no matching mach segment"
#         segment["mach_timestamps"] = ""
#         segment["mach_seg_id"] = ""
#         segment["wer"] = -1.0
# # %%
# mach_segments = deque(mach_segments)
# for segment in segments:
#     seg_text = get_seg_text(segment)
#     if seg_text != "":
#         mach_segment = mach_segments.popleft()
#         mach_seg_text = get_mach_seg_text(mach_segment)
#         wer = jiwer.wer(normalizer(seg_text), normalizer(mach_seg_text))
#         segment["mach_seg_content"] = mach_seg_text
#         segment["mach_timestamps"] = mach_segment["timestamp"]
#         segment["mach_seg_id"] = mach_segment["seg_id"]
#         segment["wer"] = wer
#     else:
#         segment["mach_seg_content"] = ""
#         segment["mach_timestamps"] = ""
#         segment["mach_seg_id"] = ""
#         segment["wer"] = 0.0
# # %%
# segments
# # %%
# mach_segments
# # %%
# for mach_segment in mach_segments:
#     segments.append(
#         {
#             "subtitle_file": "",
#             "seg_content": "",
#             "timestamp": "",
#             "id": "",
#             "seg_id": "",
#             "audio_file": "",
#             "mach_seg_content": mach_segment["seg_content"],
#             "mach_timestamps": mach_segment["timestamp"],
#             "mach_seg_id": mach_segment["seg_id"],
#             "wer": jiwer.wer(normalizer(mach_segment["seg_content"]), normalizer("")),
#         }
#     )
# %%
content = """1
00:00:04,560 --> 00:00:09,360
Who else has heard this making the rounds on\h
Instagram reels at the moment? Not only is it a\h\h

2
00:00:09,360 --> 00:00:15,920
fun song to sing along to - it's also an example\h
of how effective affirmations can be! So, today,\h\h

3
00:00:16,640 --> 00:00:19,840
get yourself a cuppa, and\h
let's talk about affirmations!

4
00:00:20,400 --> 00:00:29,840
Cheers.

5
00:00:39,360 --> 00:00:44,000
Hey everyone, welcome back to my channel, or if\h
you're just meeting me for the first time, hi!\h\h

6
00:00:44,000 --> 00:00:47,920
My name is Ebonie and this channel is here\h
to help you reconnect to your intuition,\h\h

7
00:00:47,920 --> 00:00:51,440
so you can manifest in the real world.\h
If you're interested in all things\h\h

8
00:00:51,440 --> 00:00:56,240
self-development, spirituality, and mindset, I \h
hope you enjoy today's video and consider\h\h

9
00:00:56,240 --> 00:01:01,360
subscribing, if you haven't already! In today's\h
video, I want to share with you why I believe\h\h

10
00:01:01,360 --> 00:01:07,680
positive affirmations can work so beautifully,\h
and how you can start incorporating them into\h\h

11
00:01:07,680 --> 00:01:13,040
your day. Beyond singing 'I am' by Yung Baby\h
Tate and Flo Milli! I had to look that up.

12
00:01:15,600 --> 00:01:21,680
So, what are affirmations? Affirmations are\h
essentially short positive statements. By\h\h

13
00:01:21,680 --> 00:01:26,080
repeating affirmations to yourself, you\h
are reinforcing that positive statement\h\h

14
00:01:26,080 --> 00:01:32,640
as being true to you - even if it's not quite\h
true in your real-life yet! It may sound simple,\h\h

15
00:01:32,640 --> 00:01:38,000
and even a little silly, but affirmations\h
can truly be so helpful. For example, let's\h\h

16
00:01:38,000 --> 00:01:42,240
say you're struggling with weight gain and\h
you want to lose weight, so you might state:\h\h

17
00:01:42,240 --> 00:01:48,320
"I am fit" or "I am my ideal body weight", or\h
simply "I am beautiful, I love my body." For\h\h

18
00:01:48,320 --> 00:01:52,960
affirmations to be the most effective that\h
they can be, you have to really believe it -\h\h

19
00:01:52,960 --> 00:01:58,480
even if that means suspending your disbelief for a\h
moment. Affirmations can help motivate and inspire\h\h

20
00:01:58,480 --> 00:02:03,360
you. They can help you to believe in what you're\h
trying to manifest. They can turn your negative\h\h

21
00:02:03,360 --> 00:02:09,520
thoughts into positive ones. They can help you to\h
shift your mindset. So, how do you make stating\h\h

22
00:02:09,520 --> 00:02:14,960
affirmations a daily practice? There are three\h
ways that I want to share with you today, that I\h\h

23
00:02:14,960 --> 00:02:21,440
alternate between. You can use an app, affirmation\h
cards, or write your own. For all of these methods,\h\h

24
00:02:21,440 --> 00:02:27,360
aim to repeat your chosen affirmation for five\h
minutes straight at least once a day. Sit somewhere\h\h

25
00:02:27,360 --> 00:02:33,600
quiet where you won't be disturbed, set a timer for\h
five minutes, and repeat one affirmation over and\h\h

26
00:02:33,600 --> 00:02:39,200
over again, until that timer goes off. State your\h
affirmation aloud, even if it's just a whisper.\h\h

27
00:02:39,200 --> 00:02:45,600
Focus on one affirmation at a time, and repeat that\h
same affirmation every day for as long as you feel\h\h

28
00:02:45,600 --> 00:02:52,080
you need to. You can pick an affirmation for just\h
one day, or for a week, or even for months at a time.\h\h

29
00:02:52,080 --> 00:02:57,760
Just be sure that you keep internalizing it. If\h
you feel like it's not working effectively, it\h\h

30
00:02:57,760 --> 00:03:02,160
may be time to switch it up. If you're not really\h
feeling it, it just feels like a habit, and you're\h\h

31
00:03:02,160 --> 00:03:07,440
not really getting much out of it, try changing\h
to a different affirmation. Now, you can look in\h\h

32
00:03:07,440 --> 00:03:13,280
the mirror if you like, some people do, or you can\h
even visualize at the same time. I have a video\h\h

33
00:03:13,280 --> 00:03:17,840
on visualization if you're curious to learn more\h
about it - I'll link that up in the cards as well.

34
00:03:24,400 --> 00:03:29,600
Affirmations are a form of visualization, in\h
a way. You're convincing yourself of a new\h\h

35
00:03:29,600 --> 00:03:36,320
identity, as you're wiring your mind to focus\h
on what you want to become. For example, if you\h\h

36
00:03:36,320 --> 00:03:43,040
want to become wealthy, your affirmation could be "I\h
am wealthy". You will then go about your day as your\h\h

37
00:03:43,040 --> 00:03:48,720
subconscious begins to really believe that you are\h
wealthy, even if you're not as financially abundant\h\h

38
00:03:48,720 --> 00:03:55,120
as you would like. I truly believe that all change\h
really starts with your mind. If you believe on a\h\h

39
00:03:55,120 --> 00:04:00,880
subconscious level that you are deserving of being\h
wealthy, and that soon you will be wealthy, your\h\h

40
00:04:00,880 --> 00:04:07,200
subconscious will then recognize people, things, and\h
opportunities that will actually be helpful to you,\h\h

41
00:04:07,200 --> 00:04:13,360
to get to your end goal. Affirmations highlight\h
the power of words. The language that we use has\h\h

42
00:04:13,360 --> 00:04:18,880
a huge impact on our well-being and happiness. If\h
your thoughts are negative, then you're likely to\h\h

43
00:04:18,880 --> 00:04:24,400
feel more negative. As difficult as it can\h
be, try to become aware of negative thoughts\h\h

44
00:04:24,400 --> 00:04:30,720
and flip them as they come up. Learn to catch\h
yourself on them. Hear the negativity, recognize,\h\h

45
00:04:30,720 --> 00:04:36,080
and forgive yourself for thinking it because after\h
all, you're only human and we're all capable of it\h-\h

46
00:04:36,080 --> 00:04:41,040
it's completely okay to have negative moments, by\h
all means! Once you do recognize it though, think\h\h

47
00:04:41,040 --> 00:04:47,440
the opposite. So if the negative thought is "I'm so\h
unfit and unhealthy," then what's the opposite? "I am\h\h

48
00:04:47,440 --> 00:04:53,440
fit and healthy." State that instead, even just\h
for 30 seconds, or even 10 seconds. Just state\h\h

49
00:04:53,440 --> 00:05:00,240
the opposite to your negative thought, over and\h
over again. Feel it, embody it, imagine it, visualize\h\h

50
00:05:00,240 --> 00:05:07,120
it. Imagine yourself living that as your reality.\h
What would being fit and healthy look like for you?\h\h

51
00:05:07,120 --> 00:05:13,760
For example! To make lasting change, your identity\h
needs to change first and foremost. For instance,\h\h

52
00:05:13,760 --> 00:05:18,800
if you're always saying "I just don't like\h
exercising", then your identity - the person you\h\h

53
00:05:18,800 --> 00:05:25,200
believe yourself to be, your personality - is that\h
of someone who doesn't like exercising. Chances\h\h

54
00:05:25,200 --> 00:05:31,280
are, you're likely never going to enjoy exercising,\h
if you keep telling yourself that you're not the\h\h

55
00:05:31,280 --> 00:05:38,400
kind of person who enjoys exercising. If you're\h
attributing your whole personality to that fact,\h\h

56
00:05:38,400 --> 00:05:44,960
that you have taken on as part of your identity,\h
then chances are, you'll just never like it. So,\h\h

57
00:05:44,960 --> 00:05:51,600
affirmations allow you to shift your perspective,\h
to begin changing your mind on an identity\h\h

58
00:05:51,600 --> 00:05:57,520
level, which will enable you to imagine a world\h
where you are the best version of yourself, and\h\h

59
00:05:57,520 --> 00:06:03,520
living the life that you want to. In the example\h
of not liking exercise, try stating instead "I\h\h

60
00:06:03,520 --> 00:06:11,600
enjoy exercising," "I enjoy moving my body" or "I feel\h
so great after exercising." It might sound silly,\h\h

61
00:06:11,600 --> 00:06:16,320
and if you're stating it out loud, chances\h
are you might feel a little silly too,\h\h

62
00:06:16,320 --> 00:06:22,480
but your perspective will begin to change, slowly\h
but surely. You'll begin to realize that you\h\h

63
00:06:22,480 --> 00:06:28,160
don't have to contain yourself to these negative\h
thoughts. You'll begin to realize that you're far\h\h

64
00:06:28,160 --> 00:06:33,920
more capable than you think you are. You'll begin\h
to realize that you can shift your perspective,\h\h

65
00:06:33,920 --> 00:06:40,240
and focus on building a positive mindset. You'll\h
begin to realize that so much is possible. Now,\h\h

66
00:06:40,240 --> 00:06:47,840
all that being said, here are three ways that you\h
can start incorporating affirmations into your day.

67
00:06:50,240 --> 00:06:59,680
The first method is to use an app. This is probably\h
the easiest option for most people, especially if\h\h

68
00:06:59,680 --> 00:07:06,000
you've never stated affirmations before. So, the\h
app I use is called 'I am', and you can set different\h\h

69
00:07:06,000 --> 00:07:11,280
backgrounds and things, so you can make it all\h
fancy for your phone. Here are some of the examples,\h\h

70
00:07:12,000 --> 00:07:17,200
so there are lots of different backgrounds you\h
can choose. I keep mine this sort of simple one.

71
00:07:19,520 --> 00:07:23,280
There's also a widget you can get so\h
that it comes up on your home screen.\h\h

72
00:07:24,000 --> 00:07:30,560
So, you can also, on this app, set notifications - as\h
many as you like, so you get random affirmations\h\h

73
00:07:30,560 --> 00:07:37,440
throughout your day. It's such a nice reminder to\h
stop, pause, read the affirmation, you know, really,\h\h

74
00:07:37,440 --> 00:07:42,560
as much as you can. I don't always obviously,\h
um depending on where I am or what I'm doing! \h

75
00:07:42,560 --> 00:07:47,760
But anytime I have a moment, I will read an\h
affirmation, and spend some time thinking\h\h

76
00:07:47,760 --> 00:07:53,200
about it and just repeating it, and really trying\h
to feel it - because that is what you want to do,\h\h

77
00:07:53,200 --> 00:07:57,440
you want to really be able to feel the\h
affirmation, and try to believe it as\h\h

78
00:07:57,440 --> 00:08:04,240
much as you can! So yeah, affirmations on your\h
phone is the first method I suggest you try.\h\h

79
00:08:06,160 --> 00:08:11,680
I'm also sure that there are plenty of other apps\h
out there that you can try as well. So, please do\h\h

80
00:08:11,680 --> 00:08:17,120
share any affirmation apps that you know of and\h
love, in the comments below, so we can share it\h\h

81
00:08:17,120 --> 00:08:21,920
with other people as well! And if I come across\h
any other affirmation apps that I like, I'll\h\h

82
00:08:21,920 --> 00:08:27,120
add them to the description as well, so check\h
that out whenever you're watching this video,\h\h

83
00:08:27,120 --> 00:08:30,800
because there might be more apps down below,\h
that you can you can check out for yourself.

84
00:08:33,760 --> 00:08:38,080
The second way that you can incorporate\h
affirmations into your day is by using\h\h

85
00:08:38,080 --> 00:08:46,720
an affirmation deck. I love, love, love\h
this deck from Dreamy Moons! There we go.\h\h

86
00:08:48,640 --> 00:08:53,840
These are just beautiful -\h
black and gold theme, love it!

87
00:09:01,920 --> 00:09:07,680
And actually, my seven-year-old niece loves\h
these as well. So many times now she's asked\h\h

88
00:09:07,680 --> 00:09:11,920
to look through them, and we always start off\h
with one or two cards, and end up drawing so\h\h

89
00:09:11,920 --> 00:09:17,680
many that we've gone through the entire deck. And\h
she's, she's figuring out what her favorites are,\h\h

90
00:09:17,680 --> 00:09:22,800
and really looking at them. So, she's still a\h
little young to fully recognize what they're\h\h

91
00:09:22,800 --> 00:09:26,800
about. But yeah, we've had some really interesting\h
conversations about some of the cards as well.\h\h

92
00:09:26,800 --> 00:09:32,160
So, it's a really beautiful thing to do with any\h
friends or family as well, and affirmation cards\h\h

93
00:09:32,800 --> 00:09:38,080
are just such a beautiful way that you can\h
pick one at the start of your day, and have it\h\h

94
00:09:38,080 --> 00:09:43,280
on display somewhere, like on your desk, so you can\h
look at it throughout the day as well. They're are\h\h

95
00:09:43,280 --> 00:09:48,000
beautiful idea, affirmation cards. I cannot\h
recommend these ones in particular enough. \h

96
00:09:48,000 --> 00:09:53,600
And they're the only affirmation cards I own,\h
so they are more than enough for, for anyone I\h\h

97
00:09:53,600 --> 00:09:58,800
think. So with affirmation cards, you can, like I\h
said, just draw one per day, or you can just draw\h\h

98
00:09:58,800 --> 00:10:04,720
one for your entire week, or even a month at a\h
time, however long you feel you want to focus on\h\h

99
00:10:04,720 --> 00:10:10,480
that particular affirmation for. As I said earlier,\h
you really can repeat affirmations for as long as\h\h

100
00:10:10,480 --> 00:10:22,160
you like. As long as it feels right to, basically.\h
Just go with your gut on it. The third and final\h\h

101
00:10:22,160 --> 00:10:27,680
way that you can start using affirmations\h
every single day, is by writing your own. \h

102
00:10:30,880 --> 00:10:35,840
Writing your own affirmations is truly the\h
most effective way to go about it, because\h\h

103
00:10:35,840 --> 00:10:41,360
you can really make them specific to you. The\h
app and the cards are fantastic, but they are\h\h

104
00:10:41,360 --> 00:10:48,320
more generalized. So, I do suggest starting with an\h
app because that's so easy to just do, right away.\h\h

105
00:10:48,320 --> 00:10:51,920
Um and if you can get some affirmation\h
cards, that is brilliant as well.\h\h

106
00:10:52,480 --> 00:10:57,920
But ultimately, you need to be able to write\h
your own, and get to that point where you can\h\h

107
00:10:57,920 --> 00:11:03,440
come up with an affirmation that really really\h
hits home for you. The app and the cards are\h\h

108
00:11:03,440 --> 00:11:09,120
opportunities to learn about affirmations, to\h
try out different affirmations and see what\h\h

109
00:11:09,120 --> 00:11:14,000
really works for you, and that will definitely\h
help you to write your own. When it comes to\h\h

110
00:11:14,000 --> 00:11:20,800
writing your own affirmations, try to keep them\h
as short and specific and concise as you possibly\h\h

111
00:11:20,800 --> 00:11:25,840
can. There's so much more I could go into on this\h
point, so if you'd like me to make a separate video\h\h

112
00:11:26,480 --> 00:11:31,680
about how to write your own affirmations, I would\h
be happy to, so let me know in the comments below\h\h

113
00:11:31,680 --> 00:11:41,840
if you'd like to see that as well. And if you have\h
any questions of course, leave them below as well.

114
00:11:48,000 --> 00:11:54,000
So, now I'd love to know - what do you think\h
about affirmations? Have you tried stating\h\h

115
00:11:54,000 --> 00:11:58,640
affirmations before? Do you want to\h
now that you've seen this video? I hope\h\h

116
00:11:58,640 --> 00:12:03,200
so, I hope you give it a try. I think it's\h
something that you really do have to try\h\h

117
00:12:03,200 --> 00:12:08,320
for yourself, and just have fun with it, you\h
know, you don't have to take it so seriously,\h\h

118
00:12:08,320 --> 00:12:14,000
and don't stress if you want to be able to state\h
affirmations but you don't want anyone hearing you,\h\h

119
00:12:14,000 --> 00:12:17,760
you know, if you do just want to start out\h
by saying it over and over in your head,\h\h

120
00:12:17,760 --> 00:12:21,600
then that's fine. You know, don't put too much\h
pressure on yourself, enjoy it, and do what you\h\h

121
00:12:21,600 --> 00:12:28,720
can. And while it's best to do every single day,\h
like anything really that we're trying to develop\h\h

122
00:12:28,720 --> 00:12:36,240
as a practice, that being said, you know, if you\h
miss a few days that's fine too, so just make sure\h\h

123
00:12:36,240 --> 00:12:41,600
that you're enjoying it and really feeling the\h
affirmations, even if you are just reading them. If\h\h

124
00:12:41,600 --> 00:12:46,880
you'd like to learn more about affirmations, I'll\h
have lots more info, resources, and recommendations\h\h

125
00:12:46,880 --> 00:12:51,200
over on today's blog post, linked in the\h
description below. As always, please let me\h\h

126
00:12:51,200 --> 00:12:56,400
know in the comments if you have any questions or\h
requests, or feel free to find me over on Instagram\h\h

127
00:12:56,400 --> 00:13:01,040
at Ebonie Hyland, I'd love to see you over there\h
as well. I truly hope you enjoyed today's video\h\h

128
00:13:01,040 --> 00:13:06,480
and consider subscribing, if you haven't already. I\h
post every Thursday, so be sure to ring the little\h\h

129
00:13:06,480 --> 00:13:11,360
bell to be notified, so you never miss a video.\h
Thank you so much for watching and supporting\h\h

130
00:13:11,360 --> 00:13:15,120
this channel, it really does mean the absolute\h
world to me. I hope you're having a wonderful\h\h

131
00:13:15,120 --> 00:13:33,840
day, wherever you are in the world, and I can't wait\h
to see you here again next week! So much love, bye!
"""
mach_content = """WEBVTT
Kind: captions
Language: en

00:00:04.560 --> 00:00:06.070 align:start position:0%
 
who<00:00:04.799><c> else</c><00:00:05.040><c> has</c><00:00:05.120><c> heard</c><00:00:05.359><c> this</c><00:00:05.600><c> making</c><00:00:05.920><c> the</c>

00:00:06.070 --> 00:00:06.080 align:start position:0%
who else has heard this making the
 

00:00:06.080 --> 00:00:08.390 align:start position:0%
who else has heard this making the
rounds<00:00:06.399><c> on</c><00:00:06.640><c> instagram</c><00:00:07.200><c> reels</c><00:00:07.520><c> at</c><00:00:07.600><c> the</c><00:00:07.680><c> moment</c>

00:00:08.390 --> 00:00:08.400 align:start position:0%
rounds on instagram reels at the moment
 

00:00:08.400 --> 00:00:10.629 align:start position:0%
rounds on instagram reels at the moment
not<00:00:08.639><c> only</c><00:00:08.960><c> is</c><00:00:09.120><c> it</c><00:00:09.280><c> a</c><00:00:09.360><c> fun</c><00:00:09.599><c> song</c><00:00:09.920><c> to</c><00:00:10.080><c> sing</c><00:00:10.320><c> along</c>

00:00:10.629 --> 00:00:10.639 align:start position:0%
not only is it a fun song to sing along
 

00:00:10.639 --> 00:00:13.350 align:start position:0%
not only is it a fun song to sing along
to<00:00:11.280><c> it's</c><00:00:11.519><c> also</c><00:00:11.759><c> an</c><00:00:11.920><c> example</c><00:00:12.320><c> of</c><00:00:12.480><c> how</c><00:00:12.719><c> effective</c>

00:00:13.350 --> 00:00:13.360 align:start position:0%
to it's also an example of how effective
 

00:00:13.360 --> 00:00:16.630 align:start position:0%
to it's also an example of how effective
affirmations<00:00:14.160><c> can</c><00:00:14.400><c> be</c><00:00:14.719><c> so</c><00:00:15.120><c> today</c>

00:00:16.630 --> 00:00:16.640 align:start position:0%
affirmations can be so today
 

00:00:16.640 --> 00:00:18.550 align:start position:0%
affirmations can be so today
get<00:00:16.880><c> yourself</c><00:00:17.279><c> a</c><00:00:17.359><c> cuppa</c><00:00:17.920><c> and</c><00:00:18.000><c> let's</c><00:00:18.320><c> talk</c>

00:00:18.550 --> 00:00:18.560 align:start position:0%
get yourself a cuppa and let's talk
 

00:00:18.560 --> 00:00:23.430 align:start position:0%
get yourself a cuppa and let's talk
about<00:00:19.039><c> affirmations</c>

00:00:23.430 --> 00:00:23.440 align:start position:0%
 
 

00:00:23.440 --> 00:00:28.830 align:start position:0%
 
[Music]

00:00:28.830 --> 00:00:28.840 align:start position:0%
 
 

00:00:28.840 --> 00:00:32.120 align:start position:0%
 
cheers

00:00:32.120 --> 00:00:32.130 align:start position:0%
 
 

00:00:32.130 --> 00:00:39.350 align:start position:0%
 
[Applause]

00:00:39.350 --> 00:00:39.360 align:start position:0%
 
 

00:00:39.360 --> 00:00:41.590 align:start position:0%
 
hey<00:00:39.680><c> everyone</c><00:00:40.399><c> welcome</c><00:00:40.719><c> back</c><00:00:40.879><c> to</c><00:00:41.040><c> my</c><00:00:41.280><c> channel</c>

00:00:41.590 --> 00:00:41.600 align:start position:0%
hey everyone welcome back to my channel
 

00:00:41.600 --> 00:00:42.950 align:start position:0%
hey everyone welcome back to my channel
or<00:00:41.760><c> if</c><00:00:41.920><c> you're</c><00:00:42.079><c> just</c><00:00:42.239><c> meeting</c><00:00:42.559><c> me</c><00:00:42.719><c> for</c><00:00:42.879><c> the</c>

00:00:42.950 --> 00:00:42.960 align:start position:0%
or if you're just meeting me for the
 

00:00:42.960 --> 00:00:43.990 align:start position:0%
or if you're just meeting me for the
first<00:00:43.280><c> time</c><00:00:43.600><c> hi</c>

00:00:43.990 --> 00:00:44.000 align:start position:0%
first time hi
 

00:00:44.000 --> 00:00:45.990 align:start position:0%
first time hi
my<00:00:44.160><c> name</c><00:00:44.399><c> is</c><00:00:44.559><c> ebony</c><00:00:45.200><c> and</c><00:00:45.360><c> this</c><00:00:45.600><c> channel</c><00:00:45.920><c> is</c>

00:00:45.990 --> 00:00:46.000 align:start position:0%
my name is ebony and this channel is
 

00:00:46.000 --> 00:00:47.190 align:start position:0%
my name is ebony and this channel is
here<00:00:46.160><c> to</c><00:00:46.239><c> help</c><00:00:46.399><c> you</c><00:00:46.480><c> reconnect</c><00:00:46.960><c> to</c><00:00:47.039><c> your</c>

00:00:47.190 --> 00:00:47.200 align:start position:0%
here to help you reconnect to your
 

00:00:47.200 --> 00:00:47.910 align:start position:0%
here to help you reconnect to your
intuition

00:00:47.910 --> 00:00:47.920 align:start position:0%
intuition
 

00:00:47.920 --> 00:00:50.310 align:start position:0%
intuition
so<00:00:48.160><c> you</c><00:00:48.239><c> can</c><00:00:48.399><c> manifest</c><00:00:49.039><c> in</c><00:00:49.200><c> the</c><00:00:49.360><c> real</c><00:00:49.680><c> world</c><00:00:50.160><c> if</c>

00:00:50.310 --> 00:00:50.320 align:start position:0%
so you can manifest in the real world if
 

00:00:50.320 --> 00:00:51.430 align:start position:0%
so you can manifest in the real world if
you're<00:00:50.480><c> interested</c><00:00:50.879><c> in</c><00:00:51.039><c> all</c><00:00:51.199><c> things</c>

00:00:51.430 --> 00:00:51.440 align:start position:0%
you're interested in all things
 

00:00:51.440 --> 00:00:53.270 align:start position:0%
you're interested in all things
self-development<00:00:52.239><c> spirituality</c><00:00:53.120><c> and</c>

00:00:53.270 --> 00:00:53.280 align:start position:0%
self-development spirituality and
 

00:00:53.280 --> 00:00:54.150 align:start position:0%
self-development spirituality and
mindset

00:00:54.150 --> 00:00:54.160 align:start position:0%
mindset
 

00:00:54.160 --> 00:00:55.830 align:start position:0%
mindset
i<00:00:54.320><c> hope</c><00:00:54.480><c> you</c><00:00:54.640><c> enjoyed</c><00:00:54.879><c> today's</c><00:00:55.199><c> video</c><00:00:55.680><c> and</c>

00:00:55.830 --> 00:00:55.840 align:start position:0%
i hope you enjoyed today's video and
 

00:00:55.840 --> 00:00:57.510 align:start position:0%
i hope you enjoyed today's video and
consider<00:00:56.239><c> subscribing</c><00:00:56.879><c> if</c><00:00:57.039><c> you</c><00:00:57.199><c> haven't</c>

00:00:57.510 --> 00:00:57.520 align:start position:0%
consider subscribing if you haven't
 

00:00:57.520 --> 00:00:58.069 align:start position:0%
consider subscribing if you haven't
already

00:00:58.069 --> 00:00:58.079 align:start position:0%
already
 

00:00:58.079 --> 00:00:59.830 align:start position:0%
already
in<00:00:58.160><c> today's</c><00:00:58.559><c> video</c><00:00:58.879><c> i</c><00:00:59.039><c> want</c><00:00:59.199><c> to</c><00:00:59.359><c> share</c><00:00:59.680><c> with</c>

00:00:59.830 --> 00:00:59.840 align:start position:0%
in today's video i want to share with
 

00:00:59.840 --> 00:01:01.349 align:start position:0%
in today's video i want to share with
you<00:01:00.320><c> why</c><00:01:00.640><c> i</c><00:01:00.800><c> believe</c>

00:01:01.349 --> 00:01:01.359 align:start position:0%
you why i believe
 

00:01:01.359 --> 00:01:04.229 align:start position:0%
you why i believe
positive<00:01:01.920><c> affirmations</c><00:01:02.960><c> can</c><00:01:03.199><c> work</c><00:01:03.760><c> so</c>

00:01:04.229 --> 00:01:04.239 align:start position:0%
positive affirmations can work so
 

00:01:04.239 --> 00:01:04.869 align:start position:0%
positive affirmations can work so
beautifully

00:01:04.869 --> 00:01:04.879 align:start position:0%
beautifully
 

00:01:04.879 --> 00:01:07.350 align:start position:0%
beautifully
and<00:01:05.119><c> how</c><00:01:05.519><c> you</c><00:01:05.680><c> can</c><00:01:05.920><c> start</c><00:01:06.320><c> incorporating</c><00:01:07.119><c> them</c>

00:01:07.350 --> 00:01:07.360 align:start position:0%
and how you can start incorporating them
 

00:01:07.360 --> 00:01:08.230 align:start position:0%
and how you can start incorporating them
into<00:01:07.680><c> your</c><00:01:07.920><c> day</c>

00:01:08.230 --> 00:01:08.240 align:start position:0%
into your day
 

00:01:08.240 --> 00:01:10.950 align:start position:0%
into your day
beyond<00:01:08.720><c> singing</c><00:01:09.200><c> i</c><00:01:09.520><c> am</c><00:01:09.840><c> by</c><00:01:10.080><c> young</c><00:01:10.240><c> baby</c><00:01:10.640><c> tate</c>

00:01:10.950 --> 00:01:10.960 align:start position:0%
beyond singing i am by young baby tate
 

00:01:10.960 --> 00:01:12.070 align:start position:0%
beyond singing i am by young baby tate
and<00:01:11.200><c> flo</c><00:01:11.520><c> milly</c>

00:01:12.070 --> 00:01:12.080 align:start position:0%
and flo milly
 

00:01:12.080 --> 00:01:15.590 align:start position:0%
and flo milly
i<00:01:12.240><c> had</c><00:01:12.400><c> to</c><00:01:12.479><c> look</c><00:01:12.720><c> that</c><00:01:12.880><c> up</c>

00:01:15.590 --> 00:01:15.600 align:start position:0%
 
 

00:01:15.600 --> 00:01:18.310 align:start position:0%
 
so<00:01:15.840><c> what</c><00:01:16.080><c> are</c><00:01:16.400><c> affirmations</c><00:01:17.520><c> affirmations</c>

00:01:18.310 --> 00:01:18.320 align:start position:0%
so what are affirmations affirmations
 

00:01:18.320 --> 00:01:18.630 align:start position:0%
so what are affirmations affirmations
are

00:01:18.630 --> 00:01:18.640 align:start position:0%
are
 

00:01:18.640 --> 00:01:21.670 align:start position:0%
are
essentially<00:01:19.680><c> short</c><00:01:20.240><c> positive</c><00:01:20.799><c> statements</c><00:01:21.439><c> by</c>

00:01:21.670 --> 00:01:21.680 align:start position:0%
essentially short positive statements by
 

00:01:21.680 --> 00:01:23.590 align:start position:0%
essentially short positive statements by
repeating<00:01:22.240><c> affirmations</c><00:01:22.799><c> to</c><00:01:22.960><c> yourself</c><00:01:23.439><c> you</c>

00:01:23.590 --> 00:01:23.600 align:start position:0%
repeating affirmations to yourself you
 

00:01:23.600 --> 00:01:26.070 align:start position:0%
repeating affirmations to yourself you
are<00:01:23.759><c> reinforcing</c><00:01:24.560><c> that</c><00:01:24.799><c> positive</c><00:01:25.360><c> statement</c>

00:01:26.070 --> 00:01:26.080 align:start position:0%
are reinforcing that positive statement
 

00:01:26.080 --> 00:01:28.789 align:start position:0%
are reinforcing that positive statement
as<00:01:26.320><c> being</c><00:01:26.799><c> true</c><00:01:27.200><c> to</c><00:01:27.360><c> you</c><00:01:27.759><c> even</c><00:01:28.080><c> if</c><00:01:28.240><c> it's</c><00:01:28.479><c> not</c>

00:01:28.789 --> 00:01:28.799 align:start position:0%
as being true to you even if it's not
 

00:01:28.799 --> 00:01:29.270 align:start position:0%
as being true to you even if it's not
quite

00:01:29.270 --> 00:01:29.280 align:start position:0%
quite
 

00:01:29.280 --> 00:01:32.230 align:start position:0%
quite
true<00:01:29.759><c> in</c><00:01:29.920><c> your</c><00:01:30.159><c> real</c><00:01:30.400><c> life</c><00:01:30.799><c> yet</c><00:01:31.280><c> it</c><00:01:31.520><c> may</c><00:01:31.840><c> sound</c>

00:01:32.230 --> 00:01:32.240 align:start position:0%
true in your real life yet it may sound
 

00:01:32.240 --> 00:01:34.069 align:start position:0%
true in your real life yet it may sound
simple<00:01:32.640><c> and</c><00:01:32.799><c> even</c><00:01:33.040><c> a</c><00:01:33.119><c> little</c><00:01:33.439><c> silly</c><00:01:33.840><c> but</c>

00:01:34.069 --> 00:01:34.079 align:start position:0%
simple and even a little silly but
 

00:01:34.079 --> 00:01:36.469 align:start position:0%
simple and even a little silly but
affirmations<00:01:34.799><c> can</c><00:01:35.040><c> truly</c><00:01:35.360><c> be</c><00:01:35.600><c> so</c><00:01:35.920><c> helpful</c>

00:01:36.469 --> 00:01:36.479 align:start position:0%
affirmations can truly be so helpful
 

00:01:36.479 --> 00:01:38.789 align:start position:0%
affirmations can truly be so helpful
for<00:01:36.720><c> example</c><00:01:37.759><c> let's</c><00:01:38.000><c> say</c><00:01:38.240><c> you're</c><00:01:38.400><c> struggling</c>

00:01:38.789 --> 00:01:38.799 align:start position:0%
for example let's say you're struggling
 

00:01:38.799 --> 00:01:40.630 align:start position:0%
for example let's say you're struggling
with<00:01:38.960><c> weight</c><00:01:39.200><c> gain</c><00:01:39.520><c> and</c><00:01:39.759><c> you</c><00:01:39.920><c> want</c><00:01:40.159><c> to</c><00:01:40.400><c> lose</c>

00:01:40.630 --> 00:01:40.640 align:start position:0%
with weight gain and you want to lose
 

00:01:40.640 --> 00:01:42.230 align:start position:0%
with weight gain and you want to lose
weight<00:01:40.960><c> so</c><00:01:41.200><c> you</c><00:01:41.439><c> might</c><00:01:41.759><c> state</c>

00:01:42.230 --> 00:01:42.240 align:start position:0%
weight so you might state
 

00:01:42.240 --> 00:01:45.830 align:start position:0%
weight so you might state
i<00:01:42.479><c> am</c><00:01:42.720><c> fit</c><00:01:43.119><c> or</c><00:01:43.439><c> i</c><00:01:43.759><c> am</c><00:01:44.159><c> my</c><00:01:44.399><c> ideal</c><00:01:45.040><c> body</c><00:01:45.360><c> weight</c><00:01:45.600><c> or</c>

00:01:45.830 --> 00:01:45.840 align:start position:0%
i am fit or i am my ideal body weight or
 

00:01:45.840 --> 00:01:48.310 align:start position:0%
i am fit or i am my ideal body weight or
simply<00:01:46.159><c> i</c><00:01:46.399><c> am</c><00:01:46.560><c> beautiful</c><00:01:47.040><c> i</c><00:01:47.280><c> love</c><00:01:47.520><c> my</c><00:01:47.680><c> body</c><00:01:48.000><c> for</c>

00:01:48.310 --> 00:01:48.320 align:start position:0%
simply i am beautiful i love my body for
 

00:01:48.320 --> 00:01:50.149 align:start position:0%
simply i am beautiful i love my body for
affirmations<00:01:48.960><c> to</c><00:01:49.040><c> be</c><00:01:49.200><c> the</c><00:01:49.360><c> most</c><00:01:49.680><c> effective</c>

00:01:50.149 --> 00:01:50.159 align:start position:0%
affirmations to be the most effective
 

00:01:50.159 --> 00:01:52.230 align:start position:0%
affirmations to be the most effective
that<00:01:50.240><c> they</c><00:01:50.479><c> can</c><00:01:50.799><c> be</c><00:01:51.200><c> you</c><00:01:51.439><c> have</c><00:01:51.600><c> to</c><00:01:51.840><c> really</c>

00:01:52.230 --> 00:01:52.240 align:start position:0%
that they can be you have to really
 

00:01:52.240 --> 00:01:52.950 align:start position:0%
that they can be you have to really
believe<00:01:52.720><c> it</c>

00:01:52.950 --> 00:01:52.960 align:start position:0%
believe it
 

00:01:52.960 --> 00:01:54.389 align:start position:0%
believe it
even<00:01:53.200><c> if</c><00:01:53.360><c> that</c><00:01:53.520><c> means</c><00:01:53.759><c> suspending</c><00:01:54.240><c> your</c>

00:01:54.389 --> 00:01:54.399 align:start position:0%
even if that means suspending your
 

00:01:54.399 --> 00:01:56.550 align:start position:0%
even if that means suspending your
disbelief<00:01:54.960><c> for</c><00:01:55.119><c> a</c><00:01:55.200><c> moment</c><00:01:55.600><c> affirmations</c><00:01:56.399><c> can</c>

00:01:56.550 --> 00:01:56.560 align:start position:0%
disbelief for a moment affirmations can
 

00:01:56.560 --> 00:01:57.590 align:start position:0%
disbelief for a moment affirmations can
help<00:01:56.880><c> motivate</c>

00:01:57.590 --> 00:01:57.600 align:start position:0%
help motivate
 

00:01:57.600 --> 00:01:59.670 align:start position:0%
help motivate
and<00:01:57.840><c> inspire</c><00:01:58.479><c> you</c><00:01:58.719><c> they</c><00:01:58.960><c> can</c><00:01:59.119><c> help</c><00:01:59.360><c> you</c><00:01:59.520><c> to</c>

00:01:59.670 --> 00:01:59.680 align:start position:0%
and inspire you they can help you to
 

00:01:59.680 --> 00:02:01.270 align:start position:0%
and inspire you they can help you to
believe<00:02:00.159><c> in</c><00:02:00.320><c> what</c><00:02:00.479><c> you're</c><00:02:00.719><c> trying</c><00:02:01.119><c> to</c>

00:02:01.270 --> 00:02:01.280 align:start position:0%
believe in what you're trying to
 

00:02:01.280 --> 00:02:02.069 align:start position:0%
believe in what you're trying to
manifest

00:02:02.069 --> 00:02:02.079 align:start position:0%
manifest
 

00:02:02.079 --> 00:02:03.830 align:start position:0%
manifest
they<00:02:02.320><c> can</c><00:02:02.479><c> turn</c><00:02:02.719><c> your</c><00:02:02.960><c> negative</c><00:02:03.360><c> thoughts</c>

00:02:03.830 --> 00:02:03.840 align:start position:0%
they can turn your negative thoughts
 

00:02:03.840 --> 00:02:05.350 align:start position:0%
they can turn your negative thoughts
into<00:02:04.320><c> positive</c><00:02:04.880><c> ones</c>

00:02:05.350 --> 00:02:05.360 align:start position:0%
into positive ones
 

00:02:05.360 --> 00:02:07.830 align:start position:0%
into positive ones
they<00:02:05.600><c> can</c><00:02:05.840><c> help</c><00:02:06.079><c> you</c><00:02:06.320><c> to</c><00:02:06.560><c> shift</c><00:02:07.040><c> your</c><00:02:07.280><c> mindset</c>

00:02:07.830 --> 00:02:07.840 align:start position:0%
they can help you to shift your mindset
 

00:02:07.840 --> 00:02:09.029 align:start position:0%
they can help you to shift your mindset
so<00:02:08.080><c> how</c><00:02:08.319><c> do</c><00:02:08.479><c> you</c><00:02:08.720><c> make</c>

00:02:09.029 --> 00:02:09.039 align:start position:0%
so how do you make
 

00:02:09.039 --> 00:02:11.589 align:start position:0%
so how do you make
studying<00:02:09.520><c> affirmations</c><00:02:10.239><c> a</c><00:02:10.399><c> daily</c><00:02:10.879><c> practice</c>

00:02:11.589 --> 00:02:11.599 align:start position:0%
studying affirmations a daily practice
 

00:02:11.599 --> 00:02:13.589 align:start position:0%
studying affirmations a daily practice
there<00:02:11.760><c> are</c><00:02:12.000><c> three</c><00:02:12.480><c> ways</c><00:02:12.879><c> that</c><00:02:13.120><c> i</c><00:02:13.280><c> want</c><00:02:13.520><c> to</c>

00:02:13.589 --> 00:02:13.599 align:start position:0%
there are three ways that i want to
 

00:02:13.599 --> 00:02:14.949 align:start position:0%
there are three ways that i want to
share<00:02:13.840><c> with</c><00:02:14.000><c> you</c><00:02:14.160><c> today</c><00:02:14.560><c> that</c><00:02:14.720><c> i</c>

00:02:14.949 --> 00:02:14.959 align:start position:0%
share with you today that i
 

00:02:14.959 --> 00:02:17.589 align:start position:0%
share with you today that i
alternate<00:02:15.440><c> between</c><00:02:15.920><c> you</c><00:02:16.080><c> can</c><00:02:16.319><c> use</c><00:02:16.560><c> an</c><00:02:16.800><c> app</c>

00:02:17.589 --> 00:02:17.599 align:start position:0%
alternate between you can use an app
 

00:02:17.599 --> 00:02:18.869 align:start position:0%
alternate between you can use an app
affirmation<00:02:18.319><c> cards</c>

00:02:18.869 --> 00:02:18.879 align:start position:0%
affirmation cards
 

00:02:18.879 --> 00:02:20.790 align:start position:0%
affirmation cards
or<00:02:19.120><c> write</c><00:02:19.440><c> your</c><00:02:19.680><c> own</c><00:02:19.840><c> for</c><00:02:20.080><c> all</c><00:02:20.319><c> of</c><00:02:20.480><c> these</c>

00:02:20.790 --> 00:02:20.800 align:start position:0%
or write your own for all of these
 

00:02:20.800 --> 00:02:23.030 align:start position:0%
or write your own for all of these
methods<00:02:21.440><c> aim</c><00:02:21.680><c> to</c><00:02:21.920><c> repeat</c><00:02:22.319><c> your</c><00:02:22.560><c> chosen</c>

00:02:23.030 --> 00:02:23.040 align:start position:0%
methods aim to repeat your chosen
 

00:02:23.040 --> 00:02:23.750 align:start position:0%
methods aim to repeat your chosen
affirmation

00:02:23.750 --> 00:02:23.760 align:start position:0%
affirmation
 

00:02:23.760 --> 00:02:26.229 align:start position:0%
affirmation
for<00:02:24.080><c> five</c><00:02:24.480><c> minutes</c><00:02:24.879><c> straight</c><00:02:25.360><c> at</c><00:02:25.599><c> least</c><00:02:25.920><c> once</c>

00:02:26.229 --> 00:02:26.239 align:start position:0%
for five minutes straight at least once
 

00:02:26.239 --> 00:02:28.070 align:start position:0%
for five minutes straight at least once
a<00:02:26.319><c> day</c><00:02:26.640><c> sit</c><00:02:26.879><c> somewhere</c><00:02:27.360><c> quiet</c><00:02:27.760><c> where</c><00:02:28.000><c> you</c>

00:02:28.070 --> 00:02:28.080 align:start position:0%
a day sit somewhere quiet where you
 

00:02:28.080 --> 00:02:29.350 align:start position:0%
a day sit somewhere quiet where you
won't<00:02:28.319><c> be</c><00:02:28.560><c> disturbed</c>

00:02:29.350 --> 00:02:29.360 align:start position:0%
won't be disturbed
 

00:02:29.360 --> 00:02:31.990 align:start position:0%
won't be disturbed
set<00:02:29.599><c> a</c><00:02:29.760><c> timer</c><00:02:30.160><c> for</c><00:02:30.400><c> five</c><00:02:30.720><c> minutes</c><00:02:31.280><c> and</c><00:02:31.440><c> repeat</c>

00:02:31.990 --> 00:02:32.000 align:start position:0%
set a timer for five minutes and repeat
 

00:02:32.000 --> 00:02:33.110 align:start position:0%
set a timer for five minutes and repeat
one<00:02:32.319><c> affirmation</c>

00:02:33.110 --> 00:02:33.120 align:start position:0%
one affirmation
 

00:02:33.120 --> 00:02:35.270 align:start position:0%
one affirmation
over<00:02:33.440><c> and</c><00:02:33.599><c> over</c><00:02:33.920><c> again</c><00:02:34.400><c> until</c><00:02:34.800><c> that</c><00:02:34.959><c> timer</c>

00:02:35.270 --> 00:02:35.280 align:start position:0%
over and over again until that timer
 

00:02:35.280 --> 00:02:37.430 align:start position:0%
over and over again until that timer
goes<00:02:35.519><c> off</c><00:02:35.760><c> state</c><00:02:36.080><c> your</c><00:02:36.239><c> affirmation</c><00:02:36.879><c> aloud</c>

00:02:37.430 --> 00:02:37.440 align:start position:0%
goes off state your affirmation aloud
 

00:02:37.440 --> 00:02:40.390 align:start position:0%
goes off state your affirmation aloud
even<00:02:37.680><c> if</c><00:02:37.840><c> it's</c><00:02:38.080><c> just</c><00:02:38.480><c> a</c><00:02:38.640><c> whisper</c><00:02:39.200><c> focus</c><00:02:39.680><c> on</c><00:02:40.160><c> one</c>

00:02:40.390 --> 00:02:40.400 align:start position:0%
even if it's just a whisper focus on one
 

00:02:40.400 --> 00:02:42.470 align:start position:0%
even if it's just a whisper focus on one
affirmation<00:02:41.040><c> at</c><00:02:41.200><c> a</c><00:02:41.280><c> time</c><00:02:41.680><c> and</c><00:02:41.920><c> repeat</c>

00:02:42.470 --> 00:02:42.480 align:start position:0%
affirmation at a time and repeat
 

00:02:42.480 --> 00:02:44.790 align:start position:0%
affirmation at a time and repeat
that<00:02:42.800><c> same</c><00:02:43.120><c> affirmation</c><00:02:43.840><c> every</c><00:02:44.160><c> day</c><00:02:44.480><c> for</c><00:02:44.640><c> as</c>

00:02:44.790 --> 00:02:44.800 align:start position:0%
that same affirmation every day for as
 

00:02:44.800 --> 00:02:46.470 align:start position:0%
that same affirmation every day for as
long<00:02:45.040><c> as</c><00:02:45.200><c> you</c><00:02:45.360><c> feel</c><00:02:45.599><c> you</c><00:02:45.840><c> need</c><00:02:46.080><c> to</c>

00:02:46.470 --> 00:02:46.480 align:start position:0%
long as you feel you need to
 

00:02:46.480 --> 00:02:48.949 align:start position:0%
long as you feel you need to
you<00:02:46.720><c> can</c><00:02:47.280><c> pick</c><00:02:47.519><c> an</c><00:02:47.680><c> affirmation</c><00:02:48.239><c> for</c><00:02:48.480><c> just</c><00:02:48.720><c> one</c>

00:02:48.949 --> 00:02:48.959 align:start position:0%
you can pick an affirmation for just one
 

00:02:48.959 --> 00:02:51.670 align:start position:0%
you can pick an affirmation for just one
day<00:02:49.440><c> or</c><00:02:49.599><c> for</c><00:02:49.840><c> a</c><00:02:50.000><c> week</c><00:02:50.319><c> or</c><00:02:50.480><c> even</c><00:02:50.800><c> for</c><00:02:51.040><c> months</c><00:02:51.519><c> at</c>

00:02:51.670 --> 00:02:51.680 align:start position:0%
day or for a week or even for months at
 

00:02:51.680 --> 00:02:52.070 align:start position:0%
day or for a week or even for months at
a<00:02:51.760><c> time</c>

00:02:52.070 --> 00:02:52.080 align:start position:0%
a time
 

00:02:52.080 --> 00:02:54.710 align:start position:0%
a time
just<00:02:52.319><c> be</c><00:02:52.480><c> sure</c><00:02:52.720><c> that</c><00:02:52.959><c> you</c><00:02:53.200><c> keep</c><00:02:53.680><c> internalizing</c>

00:02:54.710 --> 00:02:54.720 align:start position:0%
just be sure that you keep internalizing
 

00:02:54.720 --> 00:02:56.630 align:start position:0%
just be sure that you keep internalizing
it<00:02:54.959><c> if</c><00:02:55.120><c> you</c><00:02:55.360><c> feel</c><00:02:55.599><c> like</c><00:02:55.840><c> it's</c><00:02:56.080><c> not</c><00:02:56.239><c> working</c>

00:02:56.630 --> 00:02:56.640 align:start position:0%
it if you feel like it's not working
 

00:02:56.640 --> 00:02:57.589 align:start position:0%
it if you feel like it's not working
effectively

00:02:57.589 --> 00:02:57.599 align:start position:0%
effectively
 

00:02:57.599 --> 00:02:59.430 align:start position:0%
effectively
it<00:02:57.760><c> may</c><00:02:57.920><c> be</c><00:02:58.159><c> time</c><00:02:58.400><c> to</c><00:02:58.560><c> switch</c><00:02:58.879><c> it</c><00:02:59.040><c> up</c><00:02:59.200><c> if</c><00:02:59.280><c> you're</c>

00:02:59.430 --> 00:02:59.440 align:start position:0%
it may be time to switch it up if you're
 

00:02:59.440 --> 00:03:01.110 align:start position:0%
it may be time to switch it up if you're
not<00:02:59.599><c> really</c><00:02:59.840><c> feeling</c><00:03:00.239><c> it</c><00:03:00.400><c> it</c><00:03:00.480><c> just</c><00:03:00.640><c> feels</c><00:03:00.879><c> like</c>

00:03:01.110 --> 00:03:01.120 align:start position:0%
not really feeling it it just feels like
 

00:03:01.120 --> 00:03:01.910 align:start position:0%
not really feeling it it just feels like
a<00:03:01.280><c> habit</c>

00:03:01.910 --> 00:03:01.920 align:start position:0%
a habit
 

00:03:01.920 --> 00:03:03.110 align:start position:0%
a habit
and<00:03:02.080><c> you're</c><00:03:02.159><c> not</c><00:03:02.319><c> really</c><00:03:02.480><c> getting</c><00:03:02.720><c> much</c><00:03:02.959><c> out</c>

00:03:03.110 --> 00:03:03.120 align:start position:0%
and you're not really getting much out
 

00:03:03.120 --> 00:03:05.270 align:start position:0%
and you're not really getting much out
of<00:03:03.280><c> it</c><00:03:03.680><c> try</c><00:03:04.159><c> changing</c><00:03:04.560><c> to</c><00:03:04.720><c> a</c><00:03:04.879><c> different</c>

00:03:05.270 --> 00:03:05.280 align:start position:0%
of it try changing to a different
 

00:03:05.280 --> 00:03:06.390 align:start position:0%
of it try changing to a different
affirmation<00:03:06.000><c> now</c>

00:03:06.390 --> 00:03:06.400 align:start position:0%
affirmation now
 

00:03:06.400 --> 00:03:08.630 align:start position:0%
affirmation now
you<00:03:06.640><c> can</c><00:03:07.040><c> look</c><00:03:07.360><c> in</c><00:03:07.440><c> the</c><00:03:07.599><c> mirror</c><00:03:08.000><c> if</c><00:03:08.159><c> you</c><00:03:08.319><c> like</c>

00:03:08.630 --> 00:03:08.640 align:start position:0%
you can look in the mirror if you like
 

00:03:08.640 --> 00:03:09.670 align:start position:0%
you can look in the mirror if you like
some<00:03:08.879><c> people</c><00:03:09.200><c> do</c>

00:03:09.670 --> 00:03:09.680 align:start position:0%
some people do
 

00:03:09.680 --> 00:03:12.070 align:start position:0%
some people do
or<00:03:10.080><c> you</c><00:03:10.239><c> can</c><00:03:10.400><c> even</c><00:03:10.720><c> visualize</c><00:03:11.519><c> at</c><00:03:11.680><c> the</c><00:03:11.840><c> same</c>

00:03:12.070 --> 00:03:12.080 align:start position:0%
or you can even visualize at the same
 

00:03:12.080 --> 00:03:14.630 align:start position:0%
or you can even visualize at the same
time<00:03:12.480><c> i</c><00:03:12.560><c> have</c><00:03:12.800><c> a</c><00:03:12.879><c> video</c><00:03:13.280><c> on</c><00:03:13.440><c> visualization</c><00:03:14.480><c> if</c>

00:03:14.630 --> 00:03:14.640 align:start position:0%
time i have a video on visualization if
 

00:03:14.640 --> 00:03:16.149 align:start position:0%
time i have a video on visualization if
you're<00:03:14.959><c> curious</c><00:03:15.360><c> to</c><00:03:15.440><c> learn</c><00:03:15.680><c> more</c><00:03:15.840><c> about</c><00:03:16.080><c> it</c>

00:03:16.149 --> 00:03:16.159 align:start position:0%
you're curious to learn more about it
 

00:03:16.159 --> 00:03:17.509 align:start position:0%
you're curious to learn more about it
i'll<00:03:16.319><c> link</c><00:03:16.560><c> that</c><00:03:16.720><c> up</c><00:03:16.879><c> in</c><00:03:16.959><c> the</c><00:03:17.040><c> cards</c>

00:03:17.509 --> 00:03:17.519 align:start position:0%
i'll link that up in the cards
 

00:03:17.519 --> 00:03:21.020 align:start position:0%
i'll link that up in the cards
as<00:03:17.680><c> well</c>

00:03:21.020 --> 00:03:21.030 align:start position:0%
 
 

00:03:21.030 --> 00:03:24.390 align:start position:0%
 
[Music]

00:03:24.390 --> 00:03:24.400 align:start position:0%
[Music]
 

00:03:24.400 --> 00:03:26.949 align:start position:0%
[Music]
affirmations<00:03:25.280><c> are</c><00:03:25.519><c> a</c><00:03:25.680><c> form</c><00:03:25.920><c> of</c><00:03:26.000><c> visualization</c>

00:03:26.949 --> 00:03:26.959 align:start position:0%
affirmations are a form of visualization
 

00:03:26.959 --> 00:03:29.190 align:start position:0%
affirmations are a form of visualization
in<00:03:27.120><c> a</c><00:03:27.200><c> way</c><00:03:27.440><c> you're</c><00:03:27.680><c> convincing</c><00:03:28.319><c> yourself</c><00:03:28.879><c> of</c><00:03:29.120><c> a</c>

00:03:29.190 --> 00:03:29.200 align:start position:0%
in a way you're convincing yourself of a
 

00:03:29.200 --> 00:03:31.910 align:start position:0%
in a way you're convincing yourself of a
new<00:03:29.599><c> identity</c><00:03:30.239><c> as</c><00:03:30.480><c> you're</c><00:03:30.720><c> wiring</c><00:03:31.200><c> your</c><00:03:31.440><c> mind</c>

00:03:31.910 --> 00:03:31.920 align:start position:0%
new identity as you're wiring your mind
 

00:03:31.920 --> 00:03:32.710 align:start position:0%
new identity as you're wiring your mind
to<00:03:32.159><c> focus</c>

00:03:32.710 --> 00:03:32.720 align:start position:0%
to focus
 

00:03:32.720 --> 00:03:35.509 align:start position:0%
to focus
on<00:03:32.879><c> what</c><00:03:33.120><c> you</c><00:03:33.360><c> want</c><00:03:33.680><c> to</c><00:03:33.840><c> become</c><00:03:34.319><c> for</c><00:03:34.560><c> example</c>

00:03:35.509 --> 00:03:35.519 align:start position:0%
on what you want to become for example
 

00:03:35.519 --> 00:03:36.309 align:start position:0%
on what you want to become for example
if<00:03:35.760><c> you</c>

00:03:36.309 --> 00:03:36.319 align:start position:0%
if you
 

00:03:36.319 --> 00:03:39.030 align:start position:0%
if you
want<00:03:36.640><c> to</c><00:03:37.040><c> become</c><00:03:37.519><c> wealthy</c><00:03:38.080><c> your</c><00:03:38.319><c> affirmation</c>

00:03:39.030 --> 00:03:39.040 align:start position:0%
want to become wealthy your affirmation
 

00:03:39.040 --> 00:03:39.750 align:start position:0%
want to become wealthy your affirmation
could<00:03:39.280><c> be</c>

00:03:39.750 --> 00:03:39.760 align:start position:0%
could be
 

00:03:39.760 --> 00:03:42.229 align:start position:0%
could be
i<00:03:40.080><c> am</c><00:03:40.239><c> wealthy</c><00:03:40.799><c> you</c><00:03:41.040><c> will</c><00:03:41.280><c> then</c><00:03:41.519><c> go</c><00:03:41.760><c> about</c><00:03:42.000><c> your</c>

00:03:42.229 --> 00:03:42.239 align:start position:0%
i am wealthy you will then go about your
 

00:03:42.239 --> 00:03:44.550 align:start position:0%
i am wealthy you will then go about your
day<00:03:42.640><c> as</c><00:03:42.799><c> your</c><00:03:43.040><c> subconscious</c><00:03:43.920><c> begins</c><00:03:44.319><c> to</c>

00:03:44.550 --> 00:03:44.560 align:start position:0%
day as your subconscious begins to
 

00:03:44.560 --> 00:03:46.550 align:start position:0%
day as your subconscious begins to
really<00:03:44.879><c> believe</c><00:03:45.519><c> that</c><00:03:45.680><c> you</c><00:03:45.920><c> are</c><00:03:46.000><c> wealthy</c>

00:03:46.550 --> 00:03:46.560 align:start position:0%
really believe that you are wealthy
 

00:03:46.560 --> 00:03:48.229 align:start position:0%
really believe that you are wealthy
even<00:03:46.799><c> if</c><00:03:46.959><c> you're</c><00:03:47.200><c> not</c><00:03:47.440><c> as</c><00:03:47.599><c> financially</c>

00:03:48.229 --> 00:03:48.239 align:start position:0%
even if you're not as financially
 

00:03:48.239 --> 00:03:49.670 align:start position:0%
even if you're not as financially
abundant<00:03:48.720><c> as</c><00:03:48.879><c> you</c><00:03:49.040><c> would</c><00:03:49.280><c> like</c>

00:03:49.670 --> 00:03:49.680 align:start position:0%
abundant as you would like
 

00:03:49.680 --> 00:03:52.550 align:start position:0%
abundant as you would like
i<00:03:49.920><c> truly</c><00:03:50.239><c> believe</c><00:03:50.720><c> that</c><00:03:51.040><c> all</c><00:03:51.360><c> change</c><00:03:52.159><c> really</c>

00:03:52.550 --> 00:03:52.560 align:start position:0%
i truly believe that all change really
 

00:03:52.560 --> 00:03:53.110 align:start position:0%
i truly believe that all change really
starts

00:03:53.110 --> 00:03:53.120 align:start position:0%
starts
 

00:03:53.120 --> 00:03:55.110 align:start position:0%
starts
with<00:03:53.280><c> your</c><00:03:53.519><c> mind</c><00:03:53.840><c> if</c><00:03:54.080><c> you</c><00:03:54.239><c> believe</c><00:03:54.720><c> on</c><00:03:54.879><c> a</c>

00:03:55.110 --> 00:03:55.120 align:start position:0%
with your mind if you believe on a
 

00:03:55.120 --> 00:03:56.309 align:start position:0%
with your mind if you believe on a
subconscious<00:03:55.840><c> level</c>

00:03:56.309 --> 00:03:56.319 align:start position:0%
subconscious level
 

00:03:56.319 --> 00:03:58.630 align:start position:0%
subconscious level
that<00:03:56.480><c> you</c><00:03:56.720><c> are</c><00:03:56.879><c> deserving</c><00:03:57.599><c> of</c><00:03:57.840><c> being</c><00:03:58.080><c> wealthy</c>

00:03:58.630 --> 00:03:58.640 align:start position:0%
that you are deserving of being wealthy
 

00:03:58.640 --> 00:04:00.630 align:start position:0%
that you are deserving of being wealthy
and<00:03:58.879><c> that</c><00:03:59.200><c> soon</c><00:03:59.519><c> you</c><00:03:59.680><c> will</c><00:03:59.920><c> be</c><00:04:00.080><c> wealthy</c>

00:04:00.630 --> 00:04:00.640 align:start position:0%
and that soon you will be wealthy
 

00:04:00.640 --> 00:04:03.030 align:start position:0%
and that soon you will be wealthy
your<00:04:00.879><c> subconscious</c><00:04:01.599><c> will</c><00:04:01.840><c> then</c><00:04:02.239><c> recognize</c>

00:04:03.030 --> 00:04:03.040 align:start position:0%
your subconscious will then recognize
 

00:04:03.040 --> 00:04:04.949 align:start position:0%
your subconscious will then recognize
people<00:04:03.599><c> things</c><00:04:03.920><c> and</c><00:04:04.080><c> opportunities</c>

00:04:04.949 --> 00:04:04.959 align:start position:0%
people things and opportunities
 

00:04:04.959 --> 00:04:07.429 align:start position:0%
people things and opportunities
that<00:04:05.200><c> will</c><00:04:05.519><c> actually</c><00:04:05.920><c> be</c><00:04:06.239><c> helpful</c><00:04:06.720><c> to</c><00:04:06.879><c> you</c><00:04:07.200><c> to</c>

00:04:07.429 --> 00:04:07.439 align:start position:0%
that will actually be helpful to you to
 

00:04:07.439 --> 00:04:08.789 align:start position:0%
that will actually be helpful to you to
get<00:04:07.680><c> to</c><00:04:07.840><c> your</c><00:04:08.080><c> end</c><00:04:08.400><c> goal</c>

00:04:08.789 --> 00:04:08.799 align:start position:0%
get to your end goal
 

00:04:08.799 --> 00:04:11.509 align:start position:0%
get to your end goal
affirmations<00:04:09.760><c> highlight</c><00:04:10.319><c> the</c><00:04:10.480><c> power</c><00:04:10.959><c> of</c>

00:04:11.509 --> 00:04:11.519 align:start position:0%
affirmations highlight the power of
 

00:04:11.519 --> 00:04:12.070 align:start position:0%
affirmations highlight the power of
words

00:04:12.070 --> 00:04:12.080 align:start position:0%
words
 

00:04:12.080 --> 00:04:13.910 align:start position:0%
words
the<00:04:12.239><c> language</c><00:04:12.640><c> that</c><00:04:12.720><c> we</c><00:04:12.879><c> use</c><00:04:13.200><c> has</c><00:04:13.360><c> a</c><00:04:13.519><c> huge</c>

00:04:13.910 --> 00:04:13.920 align:start position:0%
the language that we use has a huge
 

00:04:13.920 --> 00:04:16.069 align:start position:0%
the language that we use has a huge
impact<00:04:14.400><c> on</c><00:04:14.560><c> our</c><00:04:14.720><c> well-being</c><00:04:15.360><c> and</c><00:04:15.439><c> happiness</c>

00:04:16.069 --> 00:04:16.079 align:start position:0%
impact on our well-being and happiness
 

00:04:16.079 --> 00:04:17.830 align:start position:0%
impact on our well-being and happiness
if<00:04:16.239><c> your</c><00:04:16.479><c> thoughts</c><00:04:16.720><c> are</c><00:04:16.880><c> negative</c><00:04:17.440><c> then</c>

00:04:17.830 --> 00:04:17.840 align:start position:0%
if your thoughts are negative then
 

00:04:17.840 --> 00:04:19.909 align:start position:0%
if your thoughts are negative then
you're<00:04:18.239><c> likely</c><00:04:18.639><c> to</c><00:04:18.880><c> feel</c><00:04:19.120><c> more</c><00:04:19.359><c> negative</c>

00:04:19.909 --> 00:04:19.919 align:start position:0%
you're likely to feel more negative
 

00:04:19.919 --> 00:04:22.710 align:start position:0%
you're likely to feel more negative
as<00:04:20.160><c> difficult</c><00:04:20.720><c> as</c><00:04:20.880><c> it</c><00:04:20.959><c> can</c><00:04:21.199><c> be</c><00:04:21.600><c> try</c><00:04:22.079><c> to</c><00:04:22.240><c> become</c>

00:04:22.710 --> 00:04:22.720 align:start position:0%
as difficult as it can be try to become
 

00:04:22.720 --> 00:04:24.390 align:start position:0%
as difficult as it can be try to become
aware<00:04:23.199><c> of</c><00:04:23.360><c> negative</c><00:04:23.840><c> thoughts</c>

00:04:24.390 --> 00:04:24.400 align:start position:0%
aware of negative thoughts
 

00:04:24.400 --> 00:04:26.950 align:start position:0%
aware of negative thoughts
and<00:04:24.639><c> flip</c><00:04:24.960><c> them</c><00:04:25.360><c> as</c><00:04:25.600><c> they</c><00:04:25.759><c> come</c><00:04:26.000><c> up</c><00:04:26.320><c> learn</c><00:04:26.720><c> to</c>

00:04:26.950 --> 00:04:26.960 align:start position:0%
and flip them as they come up learn to
 

00:04:26.960 --> 00:04:28.710 align:start position:0%
and flip them as they come up learn to
catch<00:04:27.360><c> yourself</c><00:04:27.919><c> on</c><00:04:28.080><c> them</c><00:04:28.320><c> hear</c><00:04:28.560><c> the</c>

00:04:28.710 --> 00:04:28.720 align:start position:0%
catch yourself on them hear the
 

00:04:28.720 --> 00:04:29.830 align:start position:0%
catch yourself on them hear the
negativity

00:04:29.830 --> 00:04:29.840 align:start position:0%
negativity
 

00:04:29.840 --> 00:04:31.830 align:start position:0%
negativity
recognize<00:04:30.720><c> and</c><00:04:30.880><c> forgive</c><00:04:31.280><c> yourself</c><00:04:31.600><c> for</c>

00:04:31.830 --> 00:04:31.840 align:start position:0%
recognize and forgive yourself for
 

00:04:31.840 --> 00:04:32.870 align:start position:0%
recognize and forgive yourself for
thinking<00:04:32.240><c> it</c><00:04:32.400><c> because</c>

00:04:32.870 --> 00:04:32.880 align:start position:0%
thinking it because
 

00:04:32.880 --> 00:04:34.870 align:start position:0%
thinking it because
after<00:04:33.280><c> all</c><00:04:33.360><c> you're</c><00:04:33.680><c> only</c><00:04:33.919><c> human</c><00:04:34.400><c> and</c><00:04:34.639><c> we're</c>

00:04:34.870 --> 00:04:34.880 align:start position:0%
after all you're only human and we're
 

00:04:34.880 --> 00:04:36.950 align:start position:0%
after all you're only human and we're
all<00:04:35.120><c> capable</c><00:04:35.680><c> of</c><00:04:35.919><c> it</c><00:04:36.080><c> it's</c><00:04:36.240><c> completely</c><00:04:36.720><c> okay</c>

00:04:36.950 --> 00:04:36.960 align:start position:0%
all capable of it it's completely okay
 

00:04:36.960 --> 00:04:38.710 align:start position:0%
all capable of it it's completely okay
to<00:04:37.120><c> have</c><00:04:37.280><c> negative</c><00:04:37.680><c> moments</c><00:04:38.160><c> by</c><00:04:38.320><c> all</c><00:04:38.479><c> means</c>

00:04:38.710 --> 00:04:38.720 align:start position:0%
to have negative moments by all means
 

00:04:38.720 --> 00:04:40.629 align:start position:0%
to have negative moments by all means
once<00:04:38.960><c> you</c><00:04:39.120><c> do</c><00:04:39.360><c> recognize</c><00:04:39.840><c> it</c><00:04:39.919><c> though</c>

00:04:40.629 --> 00:04:40.639 align:start position:0%
once you do recognize it though
 

00:04:40.639 --> 00:04:42.710 align:start position:0%
once you do recognize it though
think<00:04:41.040><c> the</c><00:04:41.360><c> opposite</c><00:04:41.840><c> so</c><00:04:42.080><c> if</c><00:04:42.240><c> the</c><00:04:42.320><c> negative</c>

00:04:42.710 --> 00:04:42.720 align:start position:0%
think the opposite so if the negative
 

00:04:42.720 --> 00:04:43.909 align:start position:0%
think the opposite so if the negative
thought<00:04:43.040><c> is</c>

00:04:43.909 --> 00:04:43.919 align:start position:0%
thought is
 

00:04:43.919 --> 00:04:46.310 align:start position:0%
thought is
i'm<00:04:44.240><c> so</c><00:04:44.479><c> unfit</c><00:04:44.960><c> and</c><00:04:45.120><c> unhealthy</c><00:04:45.840><c> then</c><00:04:46.000><c> what's</c>

00:04:46.310 --> 00:04:46.320 align:start position:0%
i'm so unfit and unhealthy then what's
 

00:04:46.320 --> 00:04:47.430 align:start position:0%
i'm so unfit and unhealthy then what's
the<00:04:46.400><c> opposite</c><00:04:46.880><c> i</c><00:04:47.120><c> am</c>

00:04:47.430 --> 00:04:47.440 align:start position:0%
the opposite i am
 

00:04:47.440 --> 00:04:50.070 align:start position:0%
the opposite i am
fit<00:04:47.759><c> and</c><00:04:47.919><c> healthy</c><00:04:48.479><c> state</c><00:04:48.960><c> that</c><00:04:49.280><c> instead</c><00:04:49.840><c> even</c>

00:04:50.070 --> 00:04:50.080 align:start position:0%
fit and healthy state that instead even
 

00:04:50.080 --> 00:04:52.469 align:start position:0%
fit and healthy state that instead even
just<00:04:50.400><c> for</c><00:04:50.639><c> 30</c><00:04:50.960><c> seconds</c><00:04:51.360><c> or</c><00:04:51.520><c> even</c><00:04:51.759><c> 10</c><00:04:52.080><c> seconds</c>

00:04:52.469 --> 00:04:52.479 align:start position:0%
just for 30 seconds or even 10 seconds
 

00:04:52.479 --> 00:04:53.430 align:start position:0%
just for 30 seconds or even 10 seconds
just<00:04:52.720><c> state</c>

00:04:53.430 --> 00:04:53.440 align:start position:0%
just state
 

00:04:53.440 --> 00:04:55.430 align:start position:0%
just state
the<00:04:53.680><c> opposite</c><00:04:54.160><c> to</c><00:04:54.320><c> your</c><00:04:54.479><c> negative</c><00:04:55.040><c> thought</c>

00:04:55.430 --> 00:04:55.440 align:start position:0%
the opposite to your negative thought
 

00:04:55.440 --> 00:04:57.270 align:start position:0%
the opposite to your negative thought
over<00:04:55.759><c> and</c><00:04:56.000><c> over</c><00:04:56.320><c> again</c>

00:04:57.270 --> 00:04:57.280 align:start position:0%
over and over again
 

00:04:57.280 --> 00:05:00.230 align:start position:0%
over and over again
feel<00:04:57.600><c> it</c><00:04:57.840><c> embody</c><00:04:58.400><c> it</c><00:04:58.720><c> imagine</c><00:04:59.280><c> it</c><00:04:59.520><c> visualize</c>

00:05:00.230 --> 00:05:00.240 align:start position:0%
feel it embody it imagine it visualize
 

00:05:00.240 --> 00:05:00.469 align:start position:0%
feel it embody it imagine it visualize
it

00:05:00.469 --> 00:05:00.479 align:start position:0%
it
 

00:05:00.479 --> 00:05:02.790 align:start position:0%
it
imagine<00:05:00.880><c> yourself</c><00:05:01.600><c> living</c><00:05:02.000><c> that</c><00:05:02.320><c> as</c><00:05:02.560><c> your</c>

00:05:02.790 --> 00:05:02.800 align:start position:0%
imagine yourself living that as your
 

00:05:02.800 --> 00:05:03.749 align:start position:0%
imagine yourself living that as your
reality

00:05:03.749 --> 00:05:03.759 align:start position:0%
reality
 

00:05:03.759 --> 00:05:05.830 align:start position:0%
reality
what<00:05:04.000><c> would</c><00:05:04.240><c> being</c><00:05:04.639><c> fit</c><00:05:04.880><c> and</c><00:05:05.039><c> healthy</c><00:05:05.600><c> look</c>

00:05:05.830 --> 00:05:05.840 align:start position:0%
what would being fit and healthy look
 

00:05:05.840 --> 00:05:07.110 align:start position:0%
what would being fit and healthy look
like<00:05:06.240><c> for</c><00:05:06.479><c> you</c>

00:05:07.110 --> 00:05:07.120 align:start position:0%
like for you
 

00:05:07.120 --> 00:05:10.070 align:start position:0%
like for you
for<00:05:07.360><c> example</c><00:05:08.160><c> to</c><00:05:08.320><c> make</c><00:05:08.639><c> lasting</c><00:05:09.199><c> change</c><00:05:09.840><c> your</c>

00:05:10.070 --> 00:05:10.080 align:start position:0%
for example to make lasting change your
 

00:05:10.080 --> 00:05:11.029 align:start position:0%
for example to make lasting change your
identity

00:05:11.029 --> 00:05:11.039 align:start position:0%
identity
 

00:05:11.039 --> 00:05:13.350 align:start position:0%
identity
needs<00:05:11.360><c> to</c><00:05:11.600><c> change</c><00:05:12.080><c> first</c><00:05:12.400><c> and</c><00:05:12.479><c> foremost</c><00:05:13.120><c> for</c>

00:05:13.350 --> 00:05:13.360 align:start position:0%
needs to change first and foremost for
 

00:05:13.360 --> 00:05:14.950 align:start position:0%
needs to change first and foremost for
instance<00:05:13.759><c> if</c><00:05:13.840><c> you're</c><00:05:14.080><c> always</c><00:05:14.479><c> saying</c>

00:05:14.950 --> 00:05:14.960 align:start position:0%
instance if you're always saying
 

00:05:14.960 --> 00:05:17.029 align:start position:0%
instance if you're always saying
i<00:05:15.199><c> just</c><00:05:15.360><c> don't</c><00:05:15.680><c> like</c><00:05:16.000><c> exercising</c><00:05:16.639><c> then</c><00:05:16.800><c> your</c>

00:05:17.029 --> 00:05:17.039 align:start position:0%
i just don't like exercising then your
 

00:05:17.039 --> 00:05:19.749 align:start position:0%
i just don't like exercising then your
identity<00:05:17.919><c> the</c><00:05:18.160><c> person</c><00:05:18.560><c> you</c><00:05:18.800><c> believe</c><00:05:19.280><c> yourself</c>

00:05:19.749 --> 00:05:19.759 align:start position:0%
identity the person you believe yourself
 

00:05:19.759 --> 00:05:20.230 align:start position:0%
identity the person you believe yourself
to<00:05:19.919><c> be</c>

00:05:20.230 --> 00:05:20.240 align:start position:0%
to be
 

00:05:20.240 --> 00:05:23.029 align:start position:0%
to be
your<00:05:20.479><c> personality</c><00:05:21.520><c> is</c><00:05:21.680><c> that</c><00:05:21.919><c> of</c><00:05:22.080><c> someone</c><00:05:22.800><c> who</c>

00:05:23.029 --> 00:05:23.039 align:start position:0%
your personality is that of someone who
 

00:05:23.039 --> 00:05:24.710 align:start position:0%
your personality is that of someone who
doesn't<00:05:23.520><c> like</c><00:05:23.919><c> exercising</c>

00:05:24.710 --> 00:05:24.720 align:start position:0%
doesn't like exercising
 

00:05:24.720 --> 00:05:27.350 align:start position:0%
doesn't like exercising
chances<00:05:25.199><c> are</c><00:05:25.520><c> you're</c><00:05:25.919><c> likely</c><00:05:26.400><c> never</c><00:05:26.720><c> going</c><00:05:27.039><c> to</c>

00:05:27.350 --> 00:05:27.360 align:start position:0%
chances are you're likely never going to
 

00:05:27.360 --> 00:05:27.990 align:start position:0%
chances are you're likely never going to
enjoy

00:05:27.990 --> 00:05:28.000 align:start position:0%
enjoy
 

00:05:28.000 --> 00:05:30.550 align:start position:0%
enjoy
exercising<00:05:28.880><c> if</c><00:05:29.039><c> you</c><00:05:29.199><c> keep</c><00:05:29.440><c> telling</c><00:05:29.759><c> yourself</c>

00:05:30.550 --> 00:05:30.560 align:start position:0%
exercising if you keep telling yourself
 

00:05:30.560 --> 00:05:32.310 align:start position:0%
exercising if you keep telling yourself
that<00:05:30.720><c> you're</c><00:05:30.960><c> not</c><00:05:31.199><c> the</c><00:05:31.280><c> kind</c><00:05:31.520><c> of</c><00:05:31.680><c> person</c><00:05:32.080><c> who</c>

00:05:32.310 --> 00:05:32.320 align:start position:0%
that you're not the kind of person who
 

00:05:32.320 --> 00:05:32.950 align:start position:0%
that you're not the kind of person who
enjoys

00:05:32.950 --> 00:05:32.960 align:start position:0%
enjoys
 

00:05:32.960 --> 00:05:35.110 align:start position:0%
enjoys
exercising<00:05:33.600><c> if</c><00:05:33.759><c> you're</c><00:05:34.000><c> attributing</c><00:05:34.800><c> your</c>

00:05:35.110 --> 00:05:35.120 align:start position:0%
exercising if you're attributing your
 

00:05:35.120 --> 00:05:36.870 align:start position:0%
exercising if you're attributing your
whole<00:05:35.440><c> personality</c>

00:05:36.870 --> 00:05:36.880 align:start position:0%
whole personality
 

00:05:36.880 --> 00:05:40.390 align:start position:0%
whole personality
to<00:05:37.199><c> that</c><00:05:37.919><c> fact</c><00:05:38.400><c> that</c><00:05:38.639><c> you</c><00:05:39.039><c> have</c><00:05:39.360><c> taken</c><00:05:39.759><c> on</c>

00:05:40.390 --> 00:05:40.400 align:start position:0%
to that fact that you have taken on
 

00:05:40.400 --> 00:05:43.110 align:start position:0%
to that fact that you have taken on
as<00:05:40.639><c> part</c><00:05:40.960><c> of</c><00:05:41.120><c> your</c><00:05:41.360><c> identity</c><00:05:42.160><c> then</c><00:05:42.720><c> chances</c>

00:05:43.110 --> 00:05:43.120 align:start position:0%
as part of your identity then chances
 

00:05:43.120 --> 00:05:44.550 align:start position:0%
as part of your identity then chances
are<00:05:43.280><c> you'll</c><00:05:43.520><c> just</c><00:05:43.759><c> never</c><00:05:44.000><c> like</c><00:05:44.320><c> it</c>

00:05:44.550 --> 00:05:44.560 align:start position:0%
are you'll just never like it
 

00:05:44.560 --> 00:05:47.350 align:start position:0%
are you'll just never like it
so<00:05:44.960><c> affirmations</c><00:05:46.000><c> allow</c><00:05:46.400><c> you</c><00:05:46.560><c> to</c><00:05:46.800><c> shift</c><00:05:47.120><c> your</c>

00:05:47.350 --> 00:05:47.360 align:start position:0%
so affirmations allow you to shift your
 

00:05:47.360 --> 00:05:48.150 align:start position:0%
so affirmations allow you to shift your
perspective

00:05:48.150 --> 00:05:48.160 align:start position:0%
perspective
 

00:05:48.160 --> 00:05:50.550 align:start position:0%
perspective
to<00:05:48.320><c> begin</c><00:05:48.720><c> changing</c><00:05:49.199><c> your</c><00:05:49.520><c> mind</c><00:05:50.160><c> on</c><00:05:50.400><c> an</c>

00:05:50.550 --> 00:05:50.560 align:start position:0%
to begin changing your mind on an
 

00:05:50.560 --> 00:05:51.590 align:start position:0%
to begin changing your mind on an
identity

00:05:51.590 --> 00:05:51.600 align:start position:0%
identity
 

00:05:51.600 --> 00:05:54.150 align:start position:0%
identity
level<00:05:52.000><c> which</c><00:05:52.240><c> will</c><00:05:52.479><c> enable</c><00:05:52.960><c> you</c><00:05:53.280><c> to</c><00:05:53.600><c> imagine</c><00:05:54.080><c> a</c>

00:05:54.150 --> 00:05:54.160 align:start position:0%
level which will enable you to imagine a
 

00:05:54.160 --> 00:05:55.430 align:start position:0%
level which will enable you to imagine a
world<00:05:54.560><c> where</c><00:05:54.800><c> you</c><00:05:55.039><c> are</c>

00:05:55.430 --> 00:05:55.440 align:start position:0%
world where you are
 

00:05:55.440 --> 00:05:57.830 align:start position:0%
world where you are
the<00:05:55.680><c> best</c><00:05:56.000><c> version</c><00:05:56.479><c> of</c><00:05:56.639><c> yourself</c><00:05:57.280><c> and</c><00:05:57.520><c> living</c>

00:05:57.830 --> 00:05:57.840 align:start position:0%
the best version of yourself and living
 

00:05:57.840 --> 00:05:59.350 align:start position:0%
the best version of yourself and living
the<00:05:58.000><c> life</c><00:05:58.240><c> that</c><00:05:58.479><c> you</c><00:05:58.720><c> want</c><00:05:58.960><c> to</c>

00:05:59.350 --> 00:05:59.360 align:start position:0%
the life that you want to
 

00:05:59.360 --> 00:06:01.670 align:start position:0%
the life that you want to
in<00:05:59.520><c> the</c><00:05:59.680><c> example</c><00:06:00.160><c> of</c><00:06:00.319><c> not</c><00:06:00.560><c> liking</c><00:06:00.960><c> exercise</c>

00:06:01.670 --> 00:06:01.680 align:start position:0%
in the example of not liking exercise
 

00:06:01.680 --> 00:06:03.189 align:start position:0%
in the example of not liking exercise
try<00:06:02.080><c> stating</c><00:06:02.639><c> instead</c>

00:06:03.189 --> 00:06:03.199 align:start position:0%
try stating instead
 

00:06:03.199 --> 00:06:06.309 align:start position:0%
try stating instead
i<00:06:03.520><c> enjoy</c><00:06:04.160><c> exercising</c><00:06:05.039><c> i</c><00:06:05.360><c> enjoy</c><00:06:05.840><c> moving</c><00:06:06.160><c> my</c>

00:06:06.309 --> 00:06:06.319 align:start position:0%
i enjoy exercising i enjoy moving my
 

00:06:06.319 --> 00:06:07.350 align:start position:0%
i enjoy exercising i enjoy moving my
body<00:06:06.720><c> or</c>

00:06:07.350 --> 00:06:07.360 align:start position:0%
body or
 

00:06:07.360 --> 00:06:10.150 align:start position:0%
body or
i<00:06:07.680><c> feel</c><00:06:08.000><c> so</c><00:06:08.240><c> great</c><00:06:08.720><c> after</c><00:06:09.120><c> exercising</c><00:06:09.919><c> it</c>

00:06:10.150 --> 00:06:10.160 align:start position:0%
i feel so great after exercising it
 

00:06:10.160 --> 00:06:11.590 align:start position:0%
i feel so great after exercising it
might<00:06:10.479><c> sound</c><00:06:10.800><c> silly</c>

00:06:11.590 --> 00:06:11.600 align:start position:0%
might sound silly
 

00:06:11.600 --> 00:06:13.270 align:start position:0%
might sound silly
and<00:06:11.840><c> if</c><00:06:11.919><c> you're</c><00:06:12.080><c> stating</c><00:06:12.479><c> it</c><00:06:12.720><c> out</c><00:06:12.880><c> loud</c>

00:06:13.270 --> 00:06:13.280 align:start position:0%
and if you're stating it out loud
 

00:06:13.280 --> 00:06:15.189 align:start position:0%
and if you're stating it out loud
chances<00:06:13.840><c> are</c><00:06:14.160><c> you</c><00:06:14.319><c> might</c><00:06:14.560><c> feel</c><00:06:14.880><c> a</c><00:06:14.960><c> little</c>

00:06:15.189 --> 00:06:15.199 align:start position:0%
chances are you might feel a little
 

00:06:15.199 --> 00:06:16.309 align:start position:0%
chances are you might feel a little
silly<00:06:15.600><c> too</c>

00:06:16.309 --> 00:06:16.319 align:start position:0%
silly too
 

00:06:16.319 --> 00:06:19.189 align:start position:0%
silly too
but<00:06:17.039><c> your</c><00:06:17.280><c> perspective</c><00:06:18.080><c> will</c><00:06:18.319><c> begin</c><00:06:18.880><c> to</c>

00:06:19.189 --> 00:06:19.199 align:start position:0%
but your perspective will begin to
 

00:06:19.199 --> 00:06:20.950 align:start position:0%
but your perspective will begin to
change<00:06:19.840><c> slowly</c><00:06:20.240><c> but</c><00:06:20.479><c> surely</c>

00:06:20.950 --> 00:06:20.960 align:start position:0%
change slowly but surely
 

00:06:20.960 --> 00:06:22.710 align:start position:0%
change slowly but surely
you'll<00:06:21.199><c> begin</c><00:06:21.520><c> to</c><00:06:21.759><c> realize</c><00:06:22.160><c> that</c><00:06:22.319><c> you</c><00:06:22.479><c> don't</c>

00:06:22.710 --> 00:06:22.720 align:start position:0%
you'll begin to realize that you don't
 

00:06:22.720 --> 00:06:24.469 align:start position:0%
you'll begin to realize that you don't
have<00:06:22.880><c> to</c><00:06:23.039><c> contain</c><00:06:23.600><c> yourself</c>

00:06:24.469 --> 00:06:24.479 align:start position:0%
have to contain yourself
 

00:06:24.479 --> 00:06:26.950 align:start position:0%
have to contain yourself
to<00:06:24.960><c> these</c><00:06:25.280><c> negative</c><00:06:25.919><c> thoughts</c><00:06:26.479><c> you'll</c><00:06:26.639><c> begin</c>

00:06:26.950 --> 00:06:26.960 align:start position:0%
to these negative thoughts you'll begin
 

00:06:26.960 --> 00:06:27.909 align:start position:0%
to these negative thoughts you'll begin
to<00:06:27.120><c> realize</c><00:06:27.440><c> that</c><00:06:27.600><c> you're</c>

00:06:27.909 --> 00:06:27.919 align:start position:0%
to realize that you're
 

00:06:27.919 --> 00:06:30.150 align:start position:0%
to realize that you're
far<00:06:28.160><c> more</c><00:06:28.479><c> capable</c><00:06:29.039><c> than</c><00:06:29.280><c> you</c><00:06:29.440><c> think</c><00:06:29.680><c> you</c><00:06:29.919><c> are</c>

00:06:30.150 --> 00:06:30.160 align:start position:0%
far more capable than you think you are
 

00:06:30.160 --> 00:06:32.230 align:start position:0%
far more capable than you think you are
you'll<00:06:30.400><c> begin</c><00:06:30.720><c> to</c><00:06:30.960><c> realize</c><00:06:31.360><c> that</c><00:06:31.680><c> you</c><00:06:31.919><c> can</c>

00:06:32.230 --> 00:06:32.240 align:start position:0%
you'll begin to realize that you can
 

00:06:32.240 --> 00:06:32.710 align:start position:0%
you'll begin to realize that you can
shift

00:06:32.710 --> 00:06:32.720 align:start position:0%
shift
 

00:06:32.720 --> 00:06:35.510 align:start position:0%
shift
your<00:06:33.039><c> perspective</c><00:06:33.919><c> and</c><00:06:34.240><c> focus</c><00:06:34.720><c> on</c><00:06:34.880><c> building</c><00:06:35.280><c> a</c>

00:06:35.510 --> 00:06:35.520 align:start position:0%
your perspective and focus on building a
 

00:06:35.520 --> 00:06:36.230 align:start position:0%
your perspective and focus on building a
positive

00:06:36.230 --> 00:06:36.240 align:start position:0%
positive
 

00:06:36.240 --> 00:06:38.710 align:start position:0%
positive
mindset<00:06:36.800><c> you'll</c><00:06:37.039><c> begin</c><00:06:37.280><c> to</c><00:06:37.440><c> realize</c><00:06:38.080><c> that</c><00:06:38.400><c> so</c>

00:06:38.710 --> 00:06:38.720 align:start position:0%
mindset you'll begin to realize that so
 

00:06:38.720 --> 00:06:39.270 align:start position:0%
mindset you'll begin to realize that so
much

00:06:39.270 --> 00:06:39.280 align:start position:0%
much
 

00:06:39.280 --> 00:06:41.830 align:start position:0%
much
is<00:06:39.440><c> possible</c><00:06:40.000><c> now</c><00:06:40.240><c> all</c><00:06:40.400><c> that</c><00:06:40.639><c> being</c><00:06:40.880><c> said</c><00:06:41.440><c> here</c>

00:06:41.830 --> 00:06:41.840 align:start position:0%
is possible now all that being said here
 

00:06:41.840 --> 00:06:44.150 align:start position:0%
is possible now all that being said here
are<00:06:42.080><c> three</c><00:06:42.400><c> ways</c><00:06:42.800><c> that</c><00:06:43.039><c> you</c><00:06:43.280><c> can</c><00:06:43.520><c> start</c>

00:06:44.150 --> 00:06:44.160 align:start position:0%
are three ways that you can start
 

00:06:44.160 --> 00:06:50.230 align:start position:0%
are three ways that you can start
incorporating<00:06:45.039><c> affirmations</c><00:06:45.919><c> into</c><00:06:46.160><c> your</c><00:06:46.840><c> day</c>

00:06:50.230 --> 00:06:50.240 align:start position:0%
 
 

00:06:50.240 --> 00:06:54.720 align:start position:0%
 
the<00:06:50.479><c> first</c><00:06:50.960><c> method</c><00:06:51.520><c> is</c><00:06:51.840><c> to</c><00:06:52.080><c> use</c><00:06:52.560><c> an</c><00:06:52.800><c> app</c>

00:06:54.720 --> 00:06:54.730 align:start position:0%
the first method is to use an app
 

00:06:54.730 --> 00:06:56.150 align:start position:0%
the first method is to use an app
[Music]

00:06:56.150 --> 00:06:56.160 align:start position:0%
[Music]
 

00:06:56.160 --> 00:06:58.309 align:start position:0%
[Music]
this<00:06:56.400><c> is</c><00:06:56.479><c> probably</c><00:06:57.120><c> the</c><00:06:57.360><c> easiest</c><00:06:57.840><c> option</c><00:06:58.160><c> for</c>

00:06:58.309 --> 00:06:58.319 align:start position:0%
this is probably the easiest option for
 

00:06:58.319 --> 00:07:00.070 align:start position:0%
this is probably the easiest option for
most<00:06:58.639><c> people</c><00:06:59.039><c> especially</c><00:06:59.599><c> if</c><00:06:59.680><c> you've</c><00:06:59.840><c> never</c>

00:07:00.070 --> 00:07:00.080 align:start position:0%
most people especially if you've never
 

00:07:00.080 --> 00:07:02.390 align:start position:0%
most people especially if you've never
stated<00:07:00.560><c> affirmations</c><00:07:01.199><c> before</c><00:07:01.680><c> so</c><00:07:01.840><c> the</c><00:07:02.000><c> app</c>

00:07:02.390 --> 00:07:02.400 align:start position:0%
stated affirmations before so the app
 

00:07:02.400 --> 00:07:05.430 align:start position:0%
stated affirmations before so the app
i<00:07:02.560><c> use</c><00:07:02.880><c> is</c><00:07:03.039><c> called</c><00:07:03.440><c> im</c><00:07:04.479><c> and</c><00:07:04.720><c> you</c><00:07:04.800><c> can</c><00:07:05.120><c> set</c>

00:07:05.430 --> 00:07:05.440 align:start position:0%
i use is called im and you can set
 

00:07:05.440 --> 00:07:07.350 align:start position:0%
i use is called im and you can set
different<00:07:06.000><c> backgrounds</c><00:07:06.720><c> and</c><00:07:06.800><c> things</c><00:07:07.120><c> so</c><00:07:07.280><c> you</c>

00:07:07.350 --> 00:07:07.360 align:start position:0%
different backgrounds and things so you
 

00:07:07.360 --> 00:07:09.510 align:start position:0%
different backgrounds and things so you
can<00:07:07.520><c> make</c><00:07:07.759><c> it</c><00:07:07.840><c> all</c><00:07:08.000><c> fancy</c><00:07:08.400><c> for</c><00:07:08.560><c> your</c><00:07:08.800><c> phone</c>

00:07:09.510 --> 00:07:09.520 align:start position:0%
can make it all fancy for your phone
 

00:07:09.520 --> 00:07:12.390 align:start position:0%
can make it all fancy for your phone
here<00:07:09.840><c> are</c><00:07:10.240><c> some</c><00:07:10.400><c> of</c><00:07:10.479><c> the</c><00:07:10.639><c> examples</c><00:07:12.000><c> so</c><00:07:12.160><c> there</c>

00:07:12.390 --> 00:07:12.400 align:start position:0%
here are some of the examples so there
 

00:07:12.400 --> 00:07:13.830 align:start position:0%
here are some of the examples so there
are<00:07:12.479><c> lots</c><00:07:12.800><c> of</c><00:07:12.880><c> different</c><00:07:13.199><c> backgrounds</c><00:07:13.759><c> you</c>

00:07:13.830 --> 00:07:13.840 align:start position:0%
are lots of different backgrounds you
 

00:07:13.840 --> 00:07:14.629 align:start position:0%
are lots of different backgrounds you
can<00:07:14.000><c> choose</c>

00:07:14.629 --> 00:07:14.639 align:start position:0%
can choose
 

00:07:14.639 --> 00:07:19.510 align:start position:0%
can choose
i<00:07:14.800><c> keep</c><00:07:15.039><c> mine</c><00:07:15.280><c> this</c><00:07:15.759><c> sort</c><00:07:16.000><c> of</c><00:07:16.160><c> simple</c><00:07:16.960><c> one</c>

00:07:19.510 --> 00:07:19.520 align:start position:0%
 
 

00:07:19.520 --> 00:07:21.430 align:start position:0%
 
there's<00:07:19.840><c> also</c><00:07:20.080><c> a</c><00:07:20.240><c> widget</c><00:07:20.639><c> you</c><00:07:20.800><c> can</c><00:07:20.960><c> get</c><00:07:21.199><c> so</c>

00:07:21.430 --> 00:07:21.440 align:start position:0%
there's also a widget you can get so
 

00:07:21.440 --> 00:07:23.990 align:start position:0%
there's also a widget you can get so
that<00:07:21.599><c> it</c><00:07:21.759><c> comes</c><00:07:22.000><c> up</c><00:07:22.160><c> on</c><00:07:22.319><c> your</c><00:07:22.560><c> home</c><00:07:22.880><c> screen</c>

00:07:23.990 --> 00:07:24.000 align:start position:0%
that it comes up on your home screen
 

00:07:24.000 --> 00:07:26.230 align:start position:0%
that it comes up on your home screen
so<00:07:24.240><c> you</c><00:07:24.400><c> can</c><00:07:24.639><c> also</c><00:07:25.199><c> on</c><00:07:25.360><c> this</c><00:07:25.599><c> app</c><00:07:25.919><c> set</c>

00:07:26.230 --> 00:07:26.240 align:start position:0%
so you can also on this app set
 

00:07:26.240 --> 00:07:27.350 align:start position:0%
so you can also on this app set
notifications

00:07:27.350 --> 00:07:27.360 align:start position:0%
notifications
 

00:07:27.360 --> 00:07:29.670 align:start position:0%
notifications
as<00:07:27.520><c> many</c><00:07:27.840><c> as</c><00:07:28.000><c> you</c><00:07:28.160><c> like</c><00:07:28.639><c> so</c><00:07:28.800><c> you</c><00:07:28.960><c> get</c><00:07:29.199><c> random</c>

00:07:29.670 --> 00:07:29.680 align:start position:0%
as many as you like so you get random
 

00:07:29.680 --> 00:07:30.550 align:start position:0%
as many as you like so you get random
affirmations

00:07:30.550 --> 00:07:30.560 align:start position:0%
affirmations
 

00:07:30.560 --> 00:07:32.629 align:start position:0%
affirmations
throughout<00:07:30.880><c> your</c><00:07:31.039><c> day</c><00:07:31.360><c> it's</c><00:07:31.680><c> such</c><00:07:32.000><c> a</c><00:07:32.240><c> nice</c>

00:07:32.629 --> 00:07:32.639 align:start position:0%
throughout your day it's such a nice
 

00:07:32.639 --> 00:07:34.230 align:start position:0%
throughout your day it's such a nice
reminder<00:07:33.360><c> to</c><00:07:33.680><c> stop</c>

00:07:34.230 --> 00:07:34.240 align:start position:0%
reminder to stop
 

00:07:34.240 --> 00:07:36.790 align:start position:0%
reminder to stop
pause<00:07:35.120><c> read</c><00:07:35.440><c> the</c><00:07:35.599><c> affirmation</c><00:07:36.479><c> you</c><00:07:36.560><c> know</c>

00:07:36.790 --> 00:07:36.800 align:start position:0%
pause read the affirmation you know
 

00:07:36.800 --> 00:07:37.430 align:start position:0%
pause read the affirmation you know
really

00:07:37.430 --> 00:07:37.440 align:start position:0%
really
 

00:07:37.440 --> 00:07:39.430 align:start position:0%
really
as<00:07:37.680><c> much</c><00:07:37.919><c> as</c><00:07:38.080><c> you</c><00:07:38.240><c> can</c><00:07:38.639><c> i</c><00:07:38.800><c> don't</c><00:07:39.120><c> always</c>

00:07:39.430 --> 00:07:39.440 align:start position:0%
as much as you can i don't always
 

00:07:39.440 --> 00:07:41.430 align:start position:0%
as much as you can i don't always
obviously<00:07:40.000><c> um</c><00:07:40.240><c> depending</c><00:07:40.720><c> on</c><00:07:40.800><c> where</c><00:07:41.039><c> i</c><00:07:41.120><c> am</c><00:07:41.360><c> or</c>

00:07:41.430 --> 00:07:41.440 align:start position:0%
obviously um depending on where i am or
 

00:07:41.440 --> 00:07:42.550 align:start position:0%
obviously um depending on where i am or
what<00:07:41.599><c> i'm</c><00:07:41.759><c> doing</c>

00:07:42.550 --> 00:07:42.560 align:start position:0%
what i'm doing
 

00:07:42.560 --> 00:07:44.869 align:start position:0%
what i'm doing
but<00:07:42.880><c> anytime</c><00:07:43.440><c> i</c><00:07:43.520><c> have</c><00:07:43.680><c> a</c><00:07:43.840><c> moment</c><00:07:44.240><c> i</c><00:07:44.400><c> will</c><00:07:44.639><c> read</c>

00:07:44.869 --> 00:07:44.879 align:start position:0%
but anytime i have a moment i will read
 

00:07:44.879 --> 00:07:45.830 align:start position:0%
but anytime i have a moment i will read
an<00:07:45.039><c> affirmation</c>

00:07:45.830 --> 00:07:45.840 align:start position:0%
an affirmation
 

00:07:45.840 --> 00:07:48.309 align:start position:0%
an affirmation
and<00:07:46.639><c> spend</c><00:07:46.879><c> some</c><00:07:47.120><c> time</c><00:07:47.440><c> thinking</c><00:07:47.759><c> about</c><00:07:48.080><c> it</c>

00:07:48.309 --> 00:07:48.319 align:start position:0%
and spend some time thinking about it
 

00:07:48.319 --> 00:07:49.830 align:start position:0%
and spend some time thinking about it
and<00:07:48.479><c> just</c><00:07:48.720><c> repeating</c><00:07:49.280><c> it</c><00:07:49.440><c> and</c>

00:07:49.830 --> 00:07:49.840 align:start position:0%
and just repeating it and
 

00:07:49.840 --> 00:07:52.150 align:start position:0%
and just repeating it and
really<00:07:50.240><c> trying</c><00:07:50.560><c> to</c><00:07:51.039><c> feel</c><00:07:51.360><c> it</c><00:07:51.599><c> because</c><00:07:51.759><c> that</c><00:07:52.000><c> is</c>

00:07:52.150 --> 00:07:52.160 align:start position:0%
really trying to feel it because that is
 

00:07:52.160 --> 00:07:53.189 align:start position:0%
really trying to feel it because that is
what<00:07:52.319><c> you</c><00:07:52.479><c> want</c><00:07:52.639><c> to</c><00:07:52.720><c> do</c>

00:07:53.189 --> 00:07:53.199 align:start position:0%
what you want to do
 

00:07:53.199 --> 00:07:55.110 align:start position:0%
what you want to do
you<00:07:53.360><c> want</c><00:07:53.520><c> to</c><00:07:53.599><c> really</c><00:07:53.919><c> be</c><00:07:54.080><c> able</c><00:07:54.240><c> to</c><00:07:54.479><c> feel</c><00:07:54.879><c> the</c>

00:07:55.110 --> 00:07:55.120 align:start position:0%
you want to really be able to feel the
 

00:07:55.120 --> 00:07:57.430 align:start position:0%
you want to really be able to feel the
affirmation<00:07:55.840><c> and</c><00:07:56.080><c> try</c><00:07:56.319><c> to</c><00:07:56.479><c> believe</c><00:07:56.960><c> it</c><00:07:57.199><c> as</c>

00:07:57.430 --> 00:07:57.440 align:start position:0%
affirmation and try to believe it as
 

00:07:57.440 --> 00:07:58.070 align:start position:0%
affirmation and try to believe it as
much<00:07:57.680><c> as</c><00:07:57.840><c> you</c>

00:07:58.070 --> 00:07:58.080 align:start position:0%
much as you
 

00:07:58.080 --> 00:08:01.909 align:start position:0%
much as you
can<00:07:58.639><c> so</c><00:07:58.879><c> yeah</c><00:07:59.840><c> affirmations</c><00:08:00.639><c> on</c><00:08:00.879><c> your</c><00:08:01.199><c> phone</c>

00:08:01.909 --> 00:08:01.919 align:start position:0%
can so yeah affirmations on your phone
 

00:08:01.919 --> 00:08:06.150 align:start position:0%
can so yeah affirmations on your phone
is<00:08:02.160><c> the</c><00:08:02.319><c> first</c><00:08:02.800><c> method</c><00:08:03.199><c> i</c><00:08:03.360><c> suggest</c><00:08:03.759><c> you</c><00:08:03.919><c> try</c>

00:08:06.150 --> 00:08:06.160 align:start position:0%
is the first method i suggest you try
 

00:08:06.160 --> 00:08:07.990 align:start position:0%
is the first method i suggest you try
i'm<00:08:06.400><c> also</c><00:08:06.720><c> sure</c><00:08:06.960><c> that</c><00:08:07.039><c> there</c><00:08:07.280><c> are</c><00:08:07.440><c> plenty</c><00:08:07.840><c> of</c>

00:08:07.990 --> 00:08:08.000 align:start position:0%
i'm also sure that there are plenty of
 

00:08:08.000 --> 00:08:09.189 align:start position:0%
i'm also sure that there are plenty of
other<00:08:08.240><c> apps</c><00:08:08.639><c> out</c><00:08:08.879><c> there</c>

00:08:09.189 --> 00:08:09.199 align:start position:0%
other apps out there
 

00:08:09.199 --> 00:08:11.670 align:start position:0%
other apps out there
that<00:08:09.360><c> you</c><00:08:09.520><c> can</c><00:08:09.680><c> try</c><00:08:10.000><c> as</c><00:08:10.160><c> well</c><00:08:10.639><c> so</c><00:08:10.960><c> please</c><00:08:11.440><c> do</c>

00:08:11.670 --> 00:08:11.680 align:start position:0%
that you can try as well so please do
 

00:08:11.680 --> 00:08:13.670 align:start position:0%
that you can try as well so please do
share<00:08:12.080><c> any</c><00:08:12.319><c> affirmation</c><00:08:12.879><c> apps</c><00:08:13.120><c> that</c><00:08:13.199><c> you</c><00:08:13.440><c> know</c>

00:08:13.670 --> 00:08:13.680 align:start position:0%
share any affirmation apps that you know
 

00:08:13.680 --> 00:08:14.710 align:start position:0%
share any affirmation apps that you know
of<00:08:13.919><c> and</c><00:08:14.080><c> love</c>

00:08:14.710 --> 00:08:14.720 align:start position:0%
of and love
 

00:08:14.720 --> 00:08:17.110 align:start position:0%
of and love
in<00:08:14.879><c> the</c><00:08:15.039><c> comments</c><00:08:15.440><c> below</c><00:08:15.919><c> so</c><00:08:16.080><c> we</c><00:08:16.240><c> can</c><00:08:16.720><c> share</c><00:08:17.039><c> it</c>

00:08:17.110 --> 00:08:17.120 align:start position:0%
in the comments below so we can share it
 

00:08:17.120 --> 00:08:18.390 align:start position:0%
in the comments below so we can share it
with<00:08:17.360><c> other</c><00:08:17.520><c> people</c><00:08:17.919><c> as</c><00:08:18.000><c> well</c>

00:08:18.390 --> 00:08:18.400 align:start position:0%
with other people as well
 

00:08:18.400 --> 00:08:20.150 align:start position:0%
with other people as well
and<00:08:18.560><c> if</c><00:08:18.800><c> i</c><00:08:19.120><c> come</c><00:08:19.280><c> across</c><00:08:19.680><c> any</c><00:08:19.919><c> other</c>

00:08:20.150 --> 00:08:20.160 align:start position:0%
and if i come across any other
 

00:08:20.160 --> 00:08:21.909 align:start position:0%
and if i come across any other
affirmation<00:08:20.800><c> apps</c><00:08:20.960><c> that</c><00:08:21.120><c> i</c><00:08:21.280><c> like</c><00:08:21.520><c> i'll</c>

00:08:21.909 --> 00:08:21.919 align:start position:0%
affirmation apps that i like i'll
 

00:08:21.919 --> 00:08:24.550 align:start position:0%
affirmation apps that i like i'll
add<00:08:22.160><c> them</c><00:08:22.319><c> to</c><00:08:22.400><c> the</c><00:08:22.560><c> description</c><00:08:23.120><c> as</c><00:08:23.280><c> well</c><00:08:23.840><c> so</c>

00:08:24.550 --> 00:08:24.560 align:start position:0%
add them to the description as well so
 

00:08:24.560 --> 00:08:25.589 align:start position:0%
add them to the description as well so
check<00:08:24.800><c> that</c><00:08:25.120><c> out</c>

00:08:25.589 --> 00:08:25.599 align:start position:0%
check that out
 

00:08:25.599 --> 00:08:27.110 align:start position:0%
check that out
whenever<00:08:26.000><c> you're</c><00:08:26.160><c> watching</c><00:08:26.479><c> this</c><00:08:26.639><c> video</c>

00:08:27.110 --> 00:08:27.120 align:start position:0%
whenever you're watching this video
 

00:08:27.120 --> 00:08:28.469 align:start position:0%
whenever you're watching this video
because<00:08:27.360><c> there</c><00:08:27.520><c> might</c><00:08:27.680><c> be</c><00:08:27.840><c> more</c><00:08:28.080><c> apps</c><00:08:28.319><c> down</c>

00:08:28.469 --> 00:08:28.479 align:start position:0%
because there might be more apps down
 

00:08:28.479 --> 00:08:29.589 align:start position:0%
because there might be more apps down
below<00:08:28.720><c> that</c><00:08:28.879><c> you</c><00:08:28.960><c> can</c>

00:08:29.589 --> 00:08:29.599 align:start position:0%
below that you can
 

00:08:29.599 --> 00:08:33.750 align:start position:0%
below that you can
you<00:08:29.680><c> can</c><00:08:29.840><c> check</c><00:08:30.080><c> out</c><00:08:30.240><c> for</c><00:08:30.400><c> yourself</c>

00:08:33.750 --> 00:08:33.760 align:start position:0%
 
 

00:08:33.760 --> 00:08:35.909 align:start position:0%
 
the<00:08:34.000><c> second</c><00:08:34.320><c> way</c><00:08:34.640><c> that</c><00:08:34.800><c> you</c><00:08:34.959><c> can</c><00:08:35.279><c> incorporate</c>

00:08:35.909 --> 00:08:35.919 align:start position:0%
the second way that you can incorporate
 

00:08:35.919 --> 00:08:38.070 align:start position:0%
the second way that you can incorporate
affirmations<00:08:36.560><c> into</c><00:08:36.719><c> your</c><00:08:36.880><c> day</c><00:08:37.200><c> is</c><00:08:37.360><c> by</c><00:08:37.599><c> using</c>

00:08:38.070 --> 00:08:38.080 align:start position:0%
affirmations into your day is by using
 

00:08:38.080 --> 00:08:39.930 align:start position:0%
affirmations into your day is by using
an<00:08:38.320><c> affirmation</c><00:08:39.039><c> deck</c>

00:08:39.930 --> 00:08:39.940 align:start position:0%
an affirmation deck
 

00:08:39.940 --> 00:08:42.550 align:start position:0%
an affirmation deck
[Music]

00:08:42.550 --> 00:08:42.560 align:start position:0%
[Music]
 

00:08:42.560 --> 00:08:45.590 align:start position:0%
[Music]
i<00:08:42.959><c> love</c><00:08:43.519><c> love</c><00:08:43.919><c> love</c><00:08:44.240><c> this</c><00:08:44.560><c> deck</c><00:08:44.880><c> from</c><00:08:45.120><c> dreamy</c>

00:08:45.590 --> 00:08:45.600 align:start position:0%
i love love love this deck from dreamy
 

00:08:45.600 --> 00:08:48.630 align:start position:0%
i love love love this deck from dreamy
moons<00:08:46.080><c> there</c><00:08:46.320><c> we</c><00:08:46.480><c> go</c>

00:08:48.630 --> 00:08:48.640 align:start position:0%
moons there we go
 

00:08:48.640 --> 00:08:51.269 align:start position:0%
moons there we go
these<00:08:49.120><c> are</c><00:08:49.360><c> just</c><00:08:49.680><c> beautiful</c><00:08:50.399><c> black</c><00:08:50.720><c> and</c><00:08:50.880><c> gold</c>

00:08:51.269 --> 00:08:51.279 align:start position:0%
these are just beautiful black and gold
 

00:08:51.279 --> 00:08:52.550 align:start position:0%
these are just beautiful black and gold
theme

00:08:52.550 --> 00:08:52.560 align:start position:0%
theme
 

00:08:52.560 --> 00:09:01.910 align:start position:0%
theme
love<00:08:52.880><c> it</c>

00:09:01.910 --> 00:09:01.920 align:start position:0%
 
 

00:09:01.920 --> 00:09:03.990 align:start position:0%
 
and<00:09:02.240><c> actually</c><00:09:02.560><c> my</c><00:09:02.800><c> seven-year-old</c><00:09:03.440><c> niece</c>

00:09:03.990 --> 00:09:04.000 align:start position:0%
and actually my seven-year-old niece
 

00:09:04.000 --> 00:09:05.430 align:start position:0%
and actually my seven-year-old niece
loves<00:09:04.320><c> these</c><00:09:04.800><c> as</c><00:09:05.040><c> well</c>

00:09:05.430 --> 00:09:05.440 align:start position:0%
loves these as well
 

00:09:05.440 --> 00:09:08.710 align:start position:0%
loves these as well
so<00:09:05.760><c> many</c><00:09:06.080><c> times</c><00:09:06.480><c> now</c><00:09:06.720><c> she's</c><00:09:07.120><c> asked</c><00:09:07.680><c> to</c><00:09:08.399><c> look</c>

00:09:08.710 --> 00:09:08.720 align:start position:0%
so many times now she's asked to look
 

00:09:08.720 --> 00:09:10.070 align:start position:0%
so many times now she's asked to look
through<00:09:08.880><c> them</c><00:09:09.200><c> and</c><00:09:09.279><c> we</c><00:09:09.360><c> always</c><00:09:09.680><c> start</c><00:09:09.920><c> off</c>

00:09:10.070 --> 00:09:10.080 align:start position:0%
through them and we always start off
 

00:09:10.080 --> 00:09:11.750 align:start position:0%
through them and we always start off
with<00:09:10.240><c> one</c><00:09:10.399><c> or</c><00:09:10.480><c> two</c><00:09:10.640><c> cards</c><00:09:10.959><c> and</c><00:09:11.040><c> end</c><00:09:11.200><c> up</c><00:09:11.360><c> drawing</c>

00:09:11.750 --> 00:09:11.760 align:start position:0%
with one or two cards and end up drawing
 

00:09:11.760 --> 00:09:13.190 align:start position:0%
with one or two cards and end up drawing
so<00:09:11.920><c> many</c><00:09:12.160><c> that</c><00:09:12.399><c> we've</c><00:09:12.560><c> gone</c><00:09:12.800><c> through</c><00:09:13.040><c> the</c>

00:09:13.190 --> 00:09:13.200 align:start position:0%
so many that we've gone through the
 

00:09:13.200 --> 00:09:13.910 align:start position:0%
so many that we've gone through the
entire

00:09:13.910 --> 00:09:13.920 align:start position:0%
entire
 

00:09:13.920 --> 00:09:16.630 align:start position:0%
entire
deck<00:09:14.320><c> and</c><00:09:14.560><c> she's</c><00:09:15.120><c> she's</c><00:09:15.519><c> figuring</c><00:09:16.000><c> out</c><00:09:16.320><c> what</c>

00:09:16.630 --> 00:09:16.640 align:start position:0%
deck and she's she's figuring out what
 

00:09:16.640 --> 00:09:18.630 align:start position:0%
deck and she's she's figuring out what
her<00:09:16.880><c> favorites</c><00:09:17.440><c> are</c><00:09:17.680><c> and</c><00:09:17.839><c> really</c><00:09:18.160><c> looking</c><00:09:18.480><c> at</c>

00:09:18.630 --> 00:09:18.640 align:start position:0%
her favorites are and really looking at
 

00:09:18.640 --> 00:09:19.190 align:start position:0%
her favorites are and really looking at
them

00:09:19.190 --> 00:09:19.200 align:start position:0%
them
 

00:09:19.200 --> 00:09:21.030 align:start position:0%
them
so<00:09:19.360><c> she's</c><00:09:19.600><c> still</c><00:09:19.760><c> a</c><00:09:19.839><c> little</c><00:09:20.160><c> young</c><00:09:20.399><c> to</c><00:09:20.640><c> fully</c>

00:09:21.030 --> 00:09:21.040 align:start position:0%
so she's still a little young to fully
 

00:09:21.040 --> 00:09:22.389 align:start position:0%
so she's still a little young to fully
recognize

00:09:22.389 --> 00:09:22.399 align:start position:0%
recognize
 

00:09:22.399 --> 00:09:23.990 align:start position:0%
recognize
what<00:09:22.560><c> they're</c><00:09:22.800><c> about</c><00:09:23.120><c> but</c><00:09:23.279><c> yeah</c><00:09:23.600><c> we've</c><00:09:23.760><c> had</c>

00:09:23.990 --> 00:09:24.000 align:start position:0%
what they're about but yeah we've had
 

00:09:24.000 --> 00:09:25.430 align:start position:0%
what they're about but yeah we've had
some<00:09:24.160><c> really</c><00:09:24.480><c> interesting</c><00:09:24.800><c> conversations</c>

00:09:25.430 --> 00:09:25.440 align:start position:0%
some really interesting conversations
 

00:09:25.440 --> 00:09:27.110 align:start position:0%
some really interesting conversations
about<00:09:25.760><c> some</c><00:09:25.920><c> of</c><00:09:26.000><c> the</c><00:09:26.080><c> cards</c><00:09:26.399><c> as</c><00:09:26.560><c> well</c><00:09:26.800><c> so</c><00:09:26.959><c> it's</c>

00:09:27.110 --> 00:09:27.120 align:start position:0%
about some of the cards as well so it's
 

00:09:27.120 --> 00:09:29.030 align:start position:0%
about some of the cards as well so it's
a<00:09:27.200><c> really</c><00:09:27.519><c> beautiful</c><00:09:27.920><c> thing</c><00:09:28.160><c> to</c><00:09:28.320><c> do</c><00:09:28.560><c> with</c>

00:09:29.030 --> 00:09:29.040 align:start position:0%
a really beautiful thing to do with
 

00:09:29.040 --> 00:09:31.190 align:start position:0%
a really beautiful thing to do with
any<00:09:29.360><c> friends</c><00:09:29.680><c> or</c><00:09:29.920><c> family</c><00:09:30.399><c> as</c><00:09:30.560><c> well</c><00:09:30.959><c> and</c>

00:09:31.190 --> 00:09:31.200 align:start position:0%
any friends or family as well and
 

00:09:31.200 --> 00:09:32.790 align:start position:0%
any friends or family as well and
affirmation<00:09:31.760><c> cards</c>

00:09:32.790 --> 00:09:32.800 align:start position:0%
affirmation cards
 

00:09:32.800 --> 00:09:35.430 align:start position:0%
affirmation cards
are<00:09:32.959><c> just</c><00:09:33.839><c> such</c><00:09:34.080><c> a</c><00:09:34.160><c> beautiful</c><00:09:34.720><c> way</c><00:09:35.040><c> that</c><00:09:35.200><c> you</c>

00:09:35.430 --> 00:09:35.440 align:start position:0%
are just such a beautiful way that you
 

00:09:35.440 --> 00:09:37.269 align:start position:0%
are just such a beautiful way that you
can<00:09:35.839><c> pick</c><00:09:36.080><c> one</c><00:09:36.320><c> at</c><00:09:36.399><c> the</c><00:09:36.560><c> start</c><00:09:36.720><c> of</c><00:09:36.880><c> your</c><00:09:37.040><c> day</c>

00:09:37.269 --> 00:09:37.279 align:start position:0%
can pick one at the start of your day
 

00:09:37.279 --> 00:09:38.070 align:start position:0%
can pick one at the start of your day
and<00:09:37.440><c> have</c><00:09:37.680><c> it</c>

00:09:38.070 --> 00:09:38.080 align:start position:0%
and have it
 

00:09:38.080 --> 00:09:40.870 align:start position:0%
and have it
on<00:09:38.240><c> display</c><00:09:38.800><c> somewhere</c><00:09:39.360><c> like</c><00:09:39.600><c> on</c><00:09:39.680><c> your</c><00:09:39.920><c> desk</c>

00:09:40.870 --> 00:09:40.880 align:start position:0%
on display somewhere like on your desk
 

00:09:40.880 --> 00:09:42.470 align:start position:0%
on display somewhere like on your desk
so<00:09:41.040><c> you</c><00:09:41.200><c> can</c><00:09:41.360><c> look</c><00:09:41.519><c> at</c><00:09:41.680><c> it</c><00:09:41.839><c> throughout</c><00:09:42.080><c> the</c><00:09:42.240><c> day</c>

00:09:42.470 --> 00:09:42.480 align:start position:0%
so you can look at it throughout the day
 

00:09:42.480 --> 00:09:43.910 align:start position:0%
so you can look at it throughout the day
as<00:09:42.640><c> well</c><00:09:42.959><c> there</c><00:09:43.120><c> are</c><00:09:43.279><c> beautiful</c>

00:09:43.910 --> 00:09:43.920 align:start position:0%
as well there are beautiful
 

00:09:43.920 --> 00:09:45.910 align:start position:0%
as well there are beautiful
idea<00:09:44.560><c> affirmation</c><00:09:45.120><c> cards</c><00:09:45.440><c> i</c><00:09:45.600><c> cannot</c>

00:09:45.910 --> 00:09:45.920 align:start position:0%
idea affirmation cards i cannot
 

00:09:45.920 --> 00:09:47.350 align:start position:0%
idea affirmation cards i cannot
recommend<00:09:46.320><c> these</c><00:09:46.480><c> ones</c><00:09:46.720><c> in</c><00:09:46.800><c> particular</c>

00:09:47.350 --> 00:09:47.360 align:start position:0%
recommend these ones in particular
 

00:09:47.360 --> 00:09:47.990 align:start position:0%
recommend these ones in particular
enough

00:09:47.990 --> 00:09:48.000 align:start position:0%
enough
 

00:09:48.000 --> 00:09:49.509 align:start position:0%
enough
and<00:09:48.080><c> they're</c><00:09:48.320><c> the</c><00:09:48.399><c> only</c><00:09:48.720><c> affirmation</c><00:09:49.120><c> cards</c><00:09:49.360><c> i</c>

00:09:49.509 --> 00:09:49.519 align:start position:0%
and they're the only affirmation cards i
 

00:09:49.519 --> 00:09:51.590 align:start position:0%
and they're the only affirmation cards i
own<00:09:49.920><c> so</c><00:09:50.720><c> they</c><00:09:51.040><c> are</c>

00:09:51.590 --> 00:09:51.600 align:start position:0%
own so they are
 

00:09:51.600 --> 00:09:53.990 align:start position:0%
own so they are
more<00:09:51.839><c> than</c><00:09:52.000><c> enough</c><00:09:52.399><c> for</c><00:09:52.959><c> for</c><00:09:53.120><c> anyone</c><00:09:53.519><c> i</c><00:09:53.600><c> think</c>

00:09:53.990 --> 00:09:54.000 align:start position:0%
more than enough for for anyone i think
 

00:09:54.000 --> 00:09:55.829 align:start position:0%
more than enough for for anyone i think
so<00:09:54.160><c> with</c><00:09:54.399><c> affirmation</c><00:09:54.880><c> cards</c><00:09:55.200><c> you</c><00:09:55.279><c> can</c><00:09:55.519><c> like</c><00:09:55.760><c> i</c>

00:09:55.829 --> 00:09:55.839 align:start position:0%
so with affirmation cards you can like i
 

00:09:55.839 --> 00:09:56.949 align:start position:0%
so with affirmation cards you can like i
said<00:09:56.080><c> just</c><00:09:56.320><c> draw</c><00:09:56.560><c> one</c>

00:09:56.949 --> 00:09:56.959 align:start position:0%
said just draw one
 

00:09:56.959 --> 00:09:59.350 align:start position:0%
said just draw one
per<00:09:57.279><c> day</c><00:09:57.920><c> or</c><00:09:58.080><c> you</c><00:09:58.240><c> can</c><00:09:58.320><c> just</c><00:09:58.480><c> draw</c><00:09:58.800><c> one</c><00:09:59.120><c> for</c>

00:09:59.350 --> 00:09:59.360 align:start position:0%
per day or you can just draw one for
 

00:09:59.360 --> 00:10:00.710 align:start position:0%
per day or you can just draw one for
your<00:09:59.600><c> entire</c><00:10:00.080><c> week</c>

00:10:00.710 --> 00:10:00.720 align:start position:0%
your entire week
 

00:10:00.720 --> 00:10:03.030 align:start position:0%
your entire week
or<00:10:01.040><c> even</c><00:10:01.360><c> a</c><00:10:01.440><c> month</c><00:10:01.760><c> at</c><00:10:01.920><c> a</c><00:10:02.000><c> time</c><00:10:02.320><c> however</c><00:10:02.800><c> long</c>

00:10:03.030 --> 00:10:03.040 align:start position:0%
or even a month at a time however long
 

00:10:03.040 --> 00:10:03.750 align:start position:0%
or even a month at a time however long
you<00:10:03.200><c> feel</c>

00:10:03.750 --> 00:10:03.760 align:start position:0%
you feel
 

00:10:03.760 --> 00:10:05.590 align:start position:0%
you feel
you<00:10:03.839><c> want</c><00:10:04.079><c> to</c><00:10:04.160><c> focus</c><00:10:04.560><c> on</c><00:10:04.720><c> that</c><00:10:04.959><c> particular</c>

00:10:05.590 --> 00:10:05.600 align:start position:0%
you want to focus on that particular
 

00:10:05.600 --> 00:10:07.750 align:start position:0%
you want to focus on that particular
affirmation<00:10:06.320><c> for</c><00:10:06.640><c> as</c><00:10:06.880><c> i</c><00:10:07.040><c> said</c><00:10:07.200><c> earlier</c><00:10:07.600><c> you</c>

00:10:07.750 --> 00:10:07.760 align:start position:0%
affirmation for as i said earlier you
 

00:10:07.760 --> 00:10:09.990 align:start position:0%
affirmation for as i said earlier you
really<00:10:08.079><c> can</c><00:10:08.399><c> repeat</c><00:10:08.880><c> affirmations</c><00:10:09.519><c> for</c><00:10:09.760><c> as</c>

00:10:09.990 --> 00:10:10.000 align:start position:0%
really can repeat affirmations for as
 

00:10:10.000 --> 00:10:11.750 align:start position:0%
really can repeat affirmations for as
long<00:10:10.320><c> as</c><00:10:10.480><c> you</c><00:10:10.720><c> like</c><00:10:10.959><c> as</c><00:10:11.040><c> long</c><00:10:11.200><c> as</c><00:10:11.360><c> it</c><00:10:11.440><c> feels</c>

00:10:11.750 --> 00:10:11.760 align:start position:0%
long as you like as long as it feels
 

00:10:11.760 --> 00:10:12.790 align:start position:0%
long as you like as long as it feels
right<00:10:12.079><c> too</c>

00:10:12.790 --> 00:10:12.800 align:start position:0%
right too
 

00:10:12.800 --> 00:10:18.260 align:start position:0%
right too
basically<00:10:13.680><c> just</c><00:10:13.920><c> go</c><00:10:14.079><c> with</c><00:10:14.240><c> your</c><00:10:14.399><c> gut</c><00:10:14.720><c> on</c><00:10:16.839><c> it</c>

00:10:18.260 --> 00:10:18.270 align:start position:0%
basically just go with your gut on it
 

00:10:18.270 --> 00:10:20.630 align:start position:0%
basically just go with your gut on it
[Music]

00:10:20.630 --> 00:10:20.640 align:start position:0%
[Music]
 

00:10:20.640 --> 00:10:23.190 align:start position:0%
[Music]
the<00:10:20.959><c> third</c><00:10:21.360><c> and</c><00:10:21.600><c> final</c><00:10:22.160><c> way</c><00:10:22.480><c> that</c><00:10:22.640><c> you</c><00:10:22.880><c> can</c>

00:10:23.190 --> 00:10:23.200 align:start position:0%
the third and final way that you can
 

00:10:23.200 --> 00:10:23.750 align:start position:0%
the third and final way that you can
start

00:10:23.750 --> 00:10:23.760 align:start position:0%
start
 

00:10:23.760 --> 00:10:26.230 align:start position:0%
start
using<00:10:24.160><c> affirmations</c><00:10:24.959><c> every</c><00:10:25.279><c> single</c><00:10:25.600><c> day</c><00:10:26.079><c> is</c>

00:10:26.230 --> 00:10:26.240 align:start position:0%
using affirmations every single day is
 

00:10:26.240 --> 00:10:28.210 align:start position:0%
using affirmations every single day is
by<00:10:26.560><c> writing</c><00:10:27.040><c> your</c><00:10:27.360><c> own</c>

00:10:28.210 --> 00:10:28.220 align:start position:0%
by writing your own
 

00:10:28.220 --> 00:10:30.870 align:start position:0%
by writing your own
[Music]

00:10:30.870 --> 00:10:30.880 align:start position:0%
[Music]
 

00:10:30.880 --> 00:10:33.430 align:start position:0%
[Music]
writing<00:10:31.279><c> your</c><00:10:31.440><c> own</c><00:10:31.680><c> affirmations</c><00:10:32.640><c> is</c><00:10:33.120><c> truly</c>

00:10:33.430 --> 00:10:33.440 align:start position:0%
writing your own affirmations is truly
 

00:10:33.440 --> 00:10:34.790 align:start position:0%
writing your own affirmations is truly
the<00:10:33.600><c> most</c><00:10:33.839><c> effective</c><00:10:34.399><c> way</c>

00:10:34.790 --> 00:10:34.800 align:start position:0%
the most effective way
 

00:10:34.800 --> 00:10:36.389 align:start position:0%
the most effective way
to<00:10:34.959><c> go</c><00:10:35.120><c> about</c><00:10:35.360><c> it</c><00:10:35.519><c> because</c><00:10:35.839><c> you</c><00:10:35.920><c> can</c><00:10:36.079><c> really</c>

00:10:36.389 --> 00:10:36.399 align:start position:0%
to go about it because you can really
 

00:10:36.399 --> 00:10:37.829 align:start position:0%
to go about it because you can really
make<00:10:36.640><c> them</c><00:10:37.040><c> specific</c>

00:10:37.829 --> 00:10:37.839 align:start position:0%
make them specific
 

00:10:37.839 --> 00:10:39.670 align:start position:0%
make them specific
to<00:10:38.000><c> you</c><00:10:38.320><c> the</c><00:10:38.480><c> app</c><00:10:38.720><c> and</c><00:10:38.800><c> the</c><00:10:38.959><c> cards</c><00:10:39.440><c> are</c>

00:10:39.670 --> 00:10:39.680 align:start position:0%
to you the app and the cards are
 

00:10:39.680 --> 00:10:41.030 align:start position:0%
to you the app and the cards are
fantastic<00:10:40.480><c> but</c><00:10:40.720><c> they</c>

00:10:41.030 --> 00:10:41.040 align:start position:0%
fantastic but they
 

00:10:41.040 --> 00:10:43.990 align:start position:0%
fantastic but they
are<00:10:41.360><c> more</c><00:10:41.920><c> generalized</c><00:10:42.640><c> so</c><00:10:42.800><c> i</c><00:10:42.959><c> do</c><00:10:43.279><c> suggest</c>

00:10:43.990 --> 00:10:44.000 align:start position:0%
are more generalized so i do suggest
 

00:10:44.000 --> 00:10:44.790 align:start position:0%
are more generalized so i do suggest
starting<00:10:44.480><c> with</c>

00:10:44.790 --> 00:10:44.800 align:start position:0%
starting with
 

00:10:44.800 --> 00:10:47.190 align:start position:0%
starting with
an<00:10:45.040><c> app</c><00:10:45.360><c> because</c><00:10:45.680><c> that's</c><00:10:46.000><c> so</c><00:10:46.240><c> easy</c><00:10:46.560><c> to</c><00:10:46.720><c> just</c><00:10:46.959><c> do</c>

00:10:47.190 --> 00:10:47.200 align:start position:0%
an app because that's so easy to just do
 

00:10:47.200 --> 00:10:48.310 align:start position:0%
an app because that's so easy to just do
right<00:10:47.440><c> away</c>

00:10:48.310 --> 00:10:48.320 align:start position:0%
right away
 

00:10:48.320 --> 00:10:50.150 align:start position:0%
right away
um<00:10:48.720><c> and</c><00:10:48.880><c> if</c><00:10:48.959><c> you</c><00:10:49.040><c> can</c><00:10:49.279><c> get</c><00:10:49.440><c> some</c><00:10:49.600><c> affirmation</c>

00:10:50.150 --> 00:10:50.160 align:start position:0%
um and if you can get some affirmation
 

00:10:50.160 --> 00:10:52.470 align:start position:0%
um and if you can get some affirmation
cards<00:10:50.560><c> that</c><00:10:50.720><c> is</c><00:10:50.880><c> brilliant</c><00:10:51.360><c> as</c><00:10:51.600><c> well</c>

00:10:52.470 --> 00:10:52.480 align:start position:0%
cards that is brilliant as well
 

00:10:52.480 --> 00:10:54.630 align:start position:0%
cards that is brilliant as well
but<00:10:52.800><c> ultimately</c><00:10:53.440><c> you</c><00:10:53.680><c> need</c><00:10:53.839><c> to</c><00:10:54.000><c> be</c><00:10:54.240><c> able</c><00:10:54.480><c> to</c>

00:10:54.630 --> 00:10:54.640 align:start position:0%
but ultimately you need to be able to
 

00:10:54.640 --> 00:10:56.150 align:start position:0%
but ultimately you need to be able to
write<00:10:54.959><c> your</c><00:10:55.200><c> own</c>

00:10:56.150 --> 00:10:56.160 align:start position:0%
write your own
 

00:10:56.160 --> 00:10:58.150 align:start position:0%
write your own
and<00:10:56.320><c> get</c><00:10:56.480><c> to</c><00:10:56.640><c> that</c><00:10:56.800><c> point</c><00:10:57.120><c> where</c><00:10:57.279><c> you</c><00:10:57.519><c> can</c><00:10:57.920><c> come</c>

00:10:58.150 --> 00:10:58.160 align:start position:0%
and get to that point where you can come
 

00:10:58.160 --> 00:10:59.590 align:start position:0%
and get to that point where you can come
up<00:10:58.240><c> with</c><00:10:58.480><c> an</c><00:10:58.640><c> affirmation</c><00:10:59.200><c> that</c>

00:10:59.590 --> 00:10:59.600 align:start position:0%
up with an affirmation that
 

00:10:59.600 --> 00:11:02.310 align:start position:0%
up with an affirmation that
really<00:11:00.160><c> really</c><00:11:00.640><c> hits</c><00:11:00.959><c> home</c><00:11:01.360><c> for</c><00:11:01.519><c> you</c><00:11:01.920><c> the</c><00:11:02.079><c> app</c>

00:11:02.310 --> 00:11:02.320 align:start position:0%
really really hits home for you the app
 

00:11:02.320 --> 00:11:03.430 align:start position:0%
really really hits home for you the app
and<00:11:02.399><c> the</c><00:11:02.560><c> cards</c><00:11:03.040><c> are</c>

00:11:03.430 --> 00:11:03.440 align:start position:0%
and the cards are
 

00:11:03.440 --> 00:11:05.829 align:start position:0%
and the cards are
opportunities<00:11:04.240><c> to</c><00:11:04.480><c> learn</c><00:11:05.360><c> about</c>

00:11:05.829 --> 00:11:05.839 align:start position:0%
opportunities to learn about
 

00:11:05.839 --> 00:11:06.710 align:start position:0%
opportunities to learn about
affirmations

00:11:06.710 --> 00:11:06.720 align:start position:0%
affirmations
 

00:11:06.720 --> 00:11:08.550 align:start position:0%
affirmations
to<00:11:06.959><c> try</c><00:11:07.279><c> out</c><00:11:07.440><c> different</c><00:11:07.839><c> affirmations</c><00:11:08.399><c> and</c>

00:11:08.550 --> 00:11:08.560 align:start position:0%
to try out different affirmations and
 

00:11:08.560 --> 00:11:10.389 align:start position:0%
to try out different affirmations and
see<00:11:08.800><c> what</c><00:11:09.120><c> really</c><00:11:09.519><c> works</c><00:11:09.920><c> for</c><00:11:10.079><c> you</c>

00:11:10.389 --> 00:11:10.399 align:start position:0%
see what really works for you
 

00:11:10.399 --> 00:11:12.470 align:start position:0%
see what really works for you
and<00:11:10.560><c> that</c><00:11:10.959><c> will</c><00:11:11.279><c> definitely</c><00:11:11.760><c> help</c><00:11:12.000><c> you</c><00:11:12.240><c> to</c>

00:11:12.470 --> 00:11:12.480 align:start position:0%
and that will definitely help you to
 

00:11:12.480 --> 00:11:13.990 align:start position:0%
and that will definitely help you to
write<00:11:12.720><c> your</c><00:11:13.040><c> own</c><00:11:13.279><c> when</c><00:11:13.440><c> it</c><00:11:13.519><c> comes</c><00:11:13.760><c> to</c>

00:11:13.990 --> 00:11:14.000 align:start position:0%
write your own when it comes to
 

00:11:14.000 --> 00:11:16.550 align:start position:0%
write your own when it comes to
writing<00:11:14.399><c> your</c><00:11:14.560><c> own</c><00:11:14.720><c> affirmations</c><00:11:16.079><c> try</c><00:11:16.399><c> to</c>

00:11:16.550 --> 00:11:16.560 align:start position:0%
writing your own affirmations try to
 

00:11:16.560 --> 00:11:17.350 align:start position:0%
writing your own affirmations try to
keep<00:11:16.800><c> them</c><00:11:17.040><c> as</c>

00:11:17.350 --> 00:11:17.360 align:start position:0%
keep them as
 

00:11:17.360 --> 00:11:20.230 align:start position:0%
keep them as
short<00:11:17.920><c> and</c><00:11:18.160><c> specific</c><00:11:18.959><c> and</c><00:11:19.120><c> concise</c><00:11:19.920><c> as</c><00:11:20.079><c> you</c>

00:11:20.230 --> 00:11:20.240 align:start position:0%
short and specific and concise as you
 

00:11:20.240 --> 00:11:21.910 align:start position:0%
short and specific and concise as you
possibly<00:11:20.800><c> can</c><00:11:21.120><c> there's</c><00:11:21.360><c> so</c><00:11:21.519><c> much</c><00:11:21.680><c> more</c><00:11:21.839><c> i</c>

00:11:21.910 --> 00:11:21.920 align:start position:0%
possibly can there's so much more i
 

00:11:21.920 --> 00:11:23.990 align:start position:0%
possibly can there's so much more i
could<00:11:22.079><c> go</c><00:11:22.320><c> into</c><00:11:22.640><c> on</c><00:11:22.880><c> this</c><00:11:23.120><c> point</c><00:11:23.519><c> so</c><00:11:23.760><c> if</c><00:11:23.839><c> you'd</c>

00:11:23.990 --> 00:11:24.000 align:start position:0%
could go into on this point so if you'd
 

00:11:24.000 --> 00:11:26.470 align:start position:0%
could go into on this point so if you'd
like<00:11:24.320><c> me</c><00:11:24.560><c> to</c><00:11:24.800><c> make</c><00:11:25.040><c> a</c><00:11:25.120><c> separate</c><00:11:25.440><c> video</c>

00:11:26.470 --> 00:11:26.480 align:start position:0%
like me to make a separate video
 

00:11:26.480 --> 00:11:29.030 align:start position:0%
like me to make a separate video
about<00:11:26.880><c> how</c><00:11:27.120><c> to</c><00:11:27.360><c> write</c><00:11:27.519><c> your</c><00:11:27.839><c> own</c><00:11:28.160><c> affirmations</c>

00:11:29.030 --> 00:11:29.040 align:start position:0%
about how to write your own affirmations
 

00:11:29.040 --> 00:11:30.870 align:start position:0%
about how to write your own affirmations
i<00:11:29.279><c> would</c><00:11:29.440><c> be</c><00:11:29.600><c> happy</c><00:11:29.920><c> to</c><00:11:30.160><c> so</c><00:11:30.399><c> let</c><00:11:30.560><c> me</c><00:11:30.640><c> know</c><00:11:30.800><c> in</c>

00:11:30.870 --> 00:11:30.880 align:start position:0%
i would be happy to so let me know in
 

00:11:30.880 --> 00:11:32.870 align:start position:0%
i would be happy to so let me know in
the<00:11:30.959><c> comments</c><00:11:31.279><c> below</c><00:11:31.680><c> if</c><00:11:31.839><c> you'd</c><00:11:32.000><c> like</c><00:11:32.320><c> to</c>

00:11:32.870 --> 00:11:32.880 align:start position:0%
the comments below if you'd like to
 

00:11:32.880 --> 00:11:35.110 align:start position:0%
the comments below if you'd like to
see<00:11:33.120><c> that</c><00:11:33.440><c> as</c><00:11:33.600><c> well</c><00:11:34.320><c> and</c><00:11:34.480><c> if</c><00:11:34.560><c> you</c><00:11:34.640><c> have</c><00:11:34.880><c> any</c>

00:11:35.110 --> 00:11:35.120 align:start position:0%
see that as well and if you have any
 

00:11:35.120 --> 00:11:35.990 align:start position:0%
see that as well and if you have any
questions<00:11:35.519><c> of</c><00:11:35.600><c> course</c>

00:11:35.990 --> 00:11:36.000 align:start position:0%
questions of course
 

00:11:36.000 --> 00:11:43.270 align:start position:0%
questions of course
leave<00:11:36.240><c> them</c><00:11:36.399><c> below</c><00:11:36.800><c> as</c><00:11:40.839><c> well</c>

00:11:43.270 --> 00:11:43.280 align:start position:0%
leave them below as well
 

00:11:43.280 --> 00:11:47.990 align:start position:0%
leave them below as well
[Music]

00:11:47.990 --> 00:11:48.000 align:start position:0%
 
 

00:11:48.000 --> 00:11:50.790 align:start position:0%
 
so<00:11:48.640><c> now</c><00:11:48.959><c> i'd</c><00:11:49.279><c> love</c><00:11:49.519><c> to</c><00:11:49.680><c> know</c><00:11:50.160><c> what</c><00:11:50.320><c> do</c><00:11:50.560><c> you</c>

00:11:50.790 --> 00:11:50.800 align:start position:0%
so now i'd love to know what do you
 

00:11:50.800 --> 00:11:51.190 align:start position:0%
so now i'd love to know what do you
think

00:11:51.190 --> 00:11:51.200 align:start position:0%
think
 

00:11:51.200 --> 00:11:53.509 align:start position:0%
think
about<00:11:51.519><c> affirmations</c><00:11:52.639><c> have</c><00:11:52.880><c> you</c><00:11:53.040><c> tried</c>

00:11:53.509 --> 00:11:53.519 align:start position:0%
about affirmations have you tried
 

00:11:53.519 --> 00:11:55.509 align:start position:0%
about affirmations have you tried
stating<00:11:54.000><c> affirmations</c><00:11:54.720><c> before</c>

00:11:55.509 --> 00:11:55.519 align:start position:0%
stating affirmations before
 

00:11:55.519 --> 00:11:57.350 align:start position:0%
stating affirmations before
do<00:11:55.680><c> you</c><00:11:55.760><c> want</c><00:11:56.000><c> to</c><00:11:56.240><c> now</c><00:11:56.639><c> that</c><00:11:56.800><c> you've</c><00:11:56.959><c> seen</c><00:11:57.200><c> this</c>

00:11:57.350 --> 00:11:57.360 align:start position:0%
do you want to now that you've seen this
 

00:11:57.360 --> 00:11:58.629 align:start position:0%
do you want to now that you've seen this
video<00:11:58.079><c> i</c><00:11:58.320><c> hope</c>

00:11:58.629 --> 00:11:58.639 align:start position:0%
video i hope
 

00:11:58.639 --> 00:12:00.710 align:start position:0%
video i hope
so<00:11:59.040><c> i</c><00:11:59.200><c> hope</c><00:11:59.360><c> you</c><00:11:59.519><c> give</c><00:11:59.680><c> it</c><00:11:59.839><c> a</c><00:11:59.920><c> try</c><00:12:00.320><c> i</c><00:12:00.399><c> think</c><00:12:00.560><c> it's</c>

00:12:00.710 --> 00:12:00.720 align:start position:0%
so i hope you give it a try i think it's
 

00:12:00.720 --> 00:12:01.910 align:start position:0%
so i hope you give it a try i think it's
something<00:12:01.120><c> that</c>

00:12:01.910 --> 00:12:01.920 align:start position:0%
something that
 

00:12:01.920 --> 00:12:04.150 align:start position:0%
something that
you<00:12:02.079><c> really</c><00:12:02.399><c> do</c><00:12:02.560><c> have</c><00:12:02.639><c> to</c><00:12:02.800><c> try</c><00:12:03.200><c> for</c><00:12:03.440><c> yourself</c>

00:12:04.150 --> 00:12:04.160 align:start position:0%
you really do have to try for yourself
 

00:12:04.160 --> 00:12:06.150 align:start position:0%
you really do have to try for yourself
and<00:12:04.399><c> just</c><00:12:04.639><c> have</c><00:12:04.959><c> fun</c><00:12:05.200><c> with</c><00:12:05.440><c> it</c><00:12:05.680><c> you</c><00:12:05.760><c> know</c><00:12:05.920><c> you</c>

00:12:06.150 --> 00:12:06.160 align:start position:0%
and just have fun with it you know you
 

00:12:06.160 --> 00:12:08.310 align:start position:0%
and just have fun with it you know you
don't<00:12:06.399><c> have</c><00:12:06.560><c> to</c><00:12:06.720><c> take</c><00:12:06.959><c> it</c><00:12:07.040><c> so</c><00:12:07.279><c> seriously</c>

00:12:08.310 --> 00:12:08.320 align:start position:0%
don't have to take it so seriously
 

00:12:08.320 --> 00:12:11.190 align:start position:0%
don't have to take it so seriously
and<00:12:08.800><c> don't</c><00:12:09.200><c> stress</c><00:12:09.680><c> if</c><00:12:10.079><c> you</c><00:12:10.639><c> want</c><00:12:10.880><c> to</c><00:12:10.959><c> be</c><00:12:11.040><c> able</c>

00:12:11.190 --> 00:12:11.200 align:start position:0%
and don't stress if you want to be able
 

00:12:11.200 --> 00:12:12.629 align:start position:0%
and don't stress if you want to be able
to<00:12:11.279><c> state</c><00:12:11.600><c> affirmations</c><00:12:12.160><c> but</c><00:12:12.320><c> you</c><00:12:12.399><c> don't</c><00:12:12.480><c> want</c>

00:12:12.629 --> 00:12:12.639 align:start position:0%
to state affirmations but you don't want
 

00:12:12.639 --> 00:12:13.990 align:start position:0%
to state affirmations but you don't want
anyone<00:12:13.120><c> hearing</c><00:12:13.440><c> you</c>

00:12:13.990 --> 00:12:14.000 align:start position:0%
anyone hearing you
 

00:12:14.000 --> 00:12:15.430 align:start position:0%
anyone hearing you
you<00:12:14.160><c> know</c><00:12:14.320><c> if</c><00:12:14.480><c> you</c><00:12:14.639><c> do</c><00:12:14.800><c> just</c><00:12:14.959><c> want</c><00:12:15.120><c> to</c><00:12:15.200><c> start</c>

00:12:15.430 --> 00:12:15.440 align:start position:0%
you know if you do just want to start
 

00:12:15.440 --> 00:12:17.430 align:start position:0%
you know if you do just want to start
out<00:12:15.600><c> by</c><00:12:16.000><c> saying</c><00:12:16.399><c> it</c><00:12:16.639><c> over</c><00:12:16.880><c> and</c><00:12:16.959><c> over</c><00:12:17.200><c> in</c><00:12:17.279><c> your</c>

00:12:17.430 --> 00:12:17.440 align:start position:0%
out by saying it over and over in your
 

00:12:17.440 --> 00:12:19.110 align:start position:0%
out by saying it over and over in your
head<00:12:17.760><c> then</c><00:12:17.920><c> that's</c><00:12:18.240><c> fine</c><00:12:18.560><c> you</c><00:12:18.639><c> know</c><00:12:18.800><c> don't</c><00:12:18.959><c> put</c>

00:12:19.110 --> 00:12:19.120 align:start position:0%
head then that's fine you know don't put
 

00:12:19.120 --> 00:12:20.230 align:start position:0%
head then that's fine you know don't put
too<00:12:19.279><c> much</c><00:12:19.440><c> pressure</c><00:12:19.680><c> on</c><00:12:19.760><c> yourself</c>

00:12:20.230 --> 00:12:20.240 align:start position:0%
too much pressure on yourself
 

00:12:20.240 --> 00:12:23.269 align:start position:0%
too much pressure on yourself
enjoy<00:12:20.720><c> it</c><00:12:20.880><c> and</c><00:12:21.040><c> do</c><00:12:21.200><c> what</c><00:12:21.360><c> you</c><00:12:21.600><c> can</c><00:12:22.320><c> and</c><00:12:22.959><c> while</c>

00:12:23.269 --> 00:12:23.279 align:start position:0%
enjoy it and do what you can and while
 

00:12:23.279 --> 00:12:24.310 align:start position:0%
enjoy it and do what you can and while
it's<00:12:23.519><c> best</c><00:12:23.839><c> to</c><00:12:24.000><c> do</c>

00:12:24.310 --> 00:12:24.320 align:start position:0%
it's best to do
 

00:12:24.320 --> 00:12:27.030 align:start position:0%
it's best to do
every<00:12:24.560><c> single</c><00:12:24.880><c> day</c><00:12:25.680><c> like</c><00:12:26.000><c> anything</c><00:12:26.560><c> really</c>

00:12:27.030 --> 00:12:27.040 align:start position:0%
every single day like anything really
 

00:12:27.040 --> 00:12:28.710 align:start position:0%
every single day like anything really
that<00:12:27.279><c> we're</c><00:12:27.440><c> trying</c><00:12:27.760><c> to</c><00:12:28.000><c> develop</c>

00:12:28.710 --> 00:12:28.720 align:start position:0%
that we're trying to develop
 

00:12:28.720 --> 00:12:31.430 align:start position:0%
that we're trying to develop
as<00:12:28.959><c> a</c><00:12:29.040><c> practice</c><00:12:29.680><c> that</c><00:12:29.920><c> being</c><00:12:30.160><c> said</c><00:12:31.040><c> you</c><00:12:31.200><c> know</c>

00:12:31.430 --> 00:12:31.440 align:start position:0%
as a practice that being said you know
 

00:12:31.440 --> 00:12:32.550 align:start position:0%
as a practice that being said you know
if<00:12:31.600><c> you</c><00:12:31.680><c> miss</c><00:12:31.920><c> a</c><00:12:32.000><c> few</c><00:12:32.160><c> days</c>

00:12:32.550 --> 00:12:32.560 align:start position:0%
if you miss a few days
 

00:12:32.560 --> 00:12:35.750 align:start position:0%
if you miss a few days
that's<00:12:32.880><c> fine</c><00:12:33.200><c> too</c><00:12:34.000><c> so</c><00:12:34.959><c> just</c>

00:12:35.750 --> 00:12:35.760 align:start position:0%
that's fine too so just
 

00:12:35.760 --> 00:12:37.590 align:start position:0%
that's fine too so just
make<00:12:36.000><c> sure</c><00:12:36.240><c> that</c><00:12:36.399><c> you're</c><00:12:36.639><c> enjoying</c><00:12:37.120><c> it</c><00:12:37.279><c> and</c>

00:12:37.590 --> 00:12:37.600 align:start position:0%
make sure that you're enjoying it and
 

00:12:37.600 --> 00:12:39.110 align:start position:0%
make sure that you're enjoying it and
really<00:12:38.480><c> feeling</c>

00:12:39.110 --> 00:12:39.120 align:start position:0%
really feeling
 

00:12:39.120 --> 00:12:40.949 align:start position:0%
really feeling
the<00:12:39.360><c> affirmations</c><00:12:40.160><c> even</c><00:12:40.320><c> if</c><00:12:40.480><c> you</c><00:12:40.560><c> are</c><00:12:40.720><c> just</c>

00:12:40.949 --> 00:12:40.959 align:start position:0%
the affirmations even if you are just
 

00:12:40.959 --> 00:12:42.389 align:start position:0%
the affirmations even if you are just
reading<00:12:41.279><c> them</c><00:12:41.440><c> if</c><00:12:41.600><c> you'd</c><00:12:41.680><c> like</c><00:12:41.839><c> to</c><00:12:42.000><c> learn</c><00:12:42.240><c> more</c>

00:12:42.389 --> 00:12:42.399 align:start position:0%
reading them if you'd like to learn more
 

00:12:42.399 --> 00:12:44.150 align:start position:0%
reading them if you'd like to learn more
about<00:12:42.639><c> affirmations</c><00:12:43.440><c> i'll</c><00:12:43.680><c> have</c>

00:12:44.150 --> 00:12:44.160 align:start position:0%
about affirmations i'll have
 

00:12:44.160 --> 00:12:46.069 align:start position:0%
about affirmations i'll have
lots<00:12:44.560><c> more</c><00:12:44.880><c> info</c><00:12:45.440><c> resources</c><00:12:46.000><c> and</c>

00:12:46.069 --> 00:12:46.079 align:start position:0%
lots more info resources and
 

00:12:46.079 --> 00:12:47.990 align:start position:0%
lots more info resources and
recommendations<00:12:46.880><c> over</c><00:12:47.120><c> on</c><00:12:47.200><c> today's</c><00:12:47.600><c> blog</c>

00:12:47.990 --> 00:12:48.000 align:start position:0%
recommendations over on today's blog
 

00:12:48.000 --> 00:12:48.550 align:start position:0%
recommendations over on today's blog
post

00:12:48.550 --> 00:12:48.560 align:start position:0%
post
 

00:12:48.560 --> 00:12:50.310 align:start position:0%
post
linked<00:12:48.800><c> in</c><00:12:48.959><c> the</c><00:12:49.040><c> description</c><00:12:49.600><c> below</c><00:12:50.079><c> as</c>

00:12:50.310 --> 00:12:50.320 align:start position:0%
linked in the description below as
 

00:12:50.320 --> 00:12:51.509 align:start position:0%
linked in the description below as
always<00:12:50.639><c> please</c><00:12:50.880><c> let</c><00:12:51.040><c> me</c><00:12:51.200><c> know</c><00:12:51.360><c> in</c><00:12:51.440><c> the</c>

00:12:51.509 --> 00:12:51.519 align:start position:0%
always please let me know in the
 

00:12:51.519 --> 00:12:53.190 align:start position:0%
always please let me know in the
comments<00:12:51.920><c> if</c><00:12:52.079><c> you</c><00:12:52.160><c> have</c><00:12:52.320><c> any</c><00:12:52.560><c> questions</c><00:12:53.040><c> or</c>

00:12:53.190 --> 00:12:53.200 align:start position:0%
comments if you have any questions or
 

00:12:53.200 --> 00:12:55.110 align:start position:0%
comments if you have any questions or
requests<00:12:53.760><c> or</c><00:12:53.920><c> feel</c><00:12:54.160><c> free</c><00:12:54.320><c> to</c><00:12:54.480><c> find</c><00:12:54.800><c> me</c>

00:12:55.110 --> 00:12:55.120 align:start position:0%
requests or feel free to find me
 

00:12:55.120 --> 00:12:57.590 align:start position:0%
requests or feel free to find me
over<00:12:55.440><c> on</c><00:12:55.680><c> instagram</c><00:12:56.399><c> at</c><00:12:56.639><c> ebony</c><00:12:56.959><c> highland</c><00:12:57.440><c> i'd</c>

00:12:57.590 --> 00:12:57.600 align:start position:0%
over on instagram at ebony highland i'd
 

00:12:57.600 --> 00:12:59.190 align:start position:0%
over on instagram at ebony highland i'd
love<00:12:57.839><c> to</c><00:12:57.920><c> see</c><00:12:58.079><c> you</c><00:12:58.240><c> over</c><00:12:58.480><c> there</c><00:12:58.720><c> as</c><00:12:58.880><c> well</c>

00:12:59.190 --> 00:12:59.200 align:start position:0%
love to see you over there as well
 

00:12:59.200 --> 00:13:01.030 align:start position:0%
love to see you over there as well
i<00:12:59.360><c> truly</c><00:12:59.680><c> hope</c><00:12:59.920><c> you</c><00:13:00.000><c> enjoyed</c><00:13:00.320><c> today's</c><00:13:00.639><c> video</c>

00:13:01.030 --> 00:13:01.040 align:start position:0%
i truly hope you enjoyed today's video
 

00:13:01.040 --> 00:13:02.949 align:start position:0%
i truly hope you enjoyed today's video
and<00:13:01.200><c> consider</c><00:13:01.680><c> subscribing</c><00:13:02.240><c> if</c><00:13:02.399><c> you</c><00:13:02.560><c> haven't</c>

00:13:02.949 --> 00:13:02.959 align:start position:0%
and consider subscribing if you haven't
 

00:13:02.959 --> 00:13:03.750 align:start position:0%
and consider subscribing if you haven't
already

00:13:03.750 --> 00:13:03.760 align:start position:0%
already
 

00:13:03.760 --> 00:13:06.150 align:start position:0%
already
i<00:13:03.920><c> post</c><00:13:04.399><c> every</c><00:13:04.720><c> thursday</c><00:13:05.360><c> so</c><00:13:05.519><c> be</c><00:13:05.680><c> sure</c><00:13:05.839><c> to</c><00:13:06.000><c> ring</c>

00:13:06.150 --> 00:13:06.160 align:start position:0%
i post every thursday so be sure to ring
 

00:13:06.160 --> 00:13:07.910 align:start position:0%
i post every thursday so be sure to ring
the<00:13:06.240><c> little</c><00:13:06.480><c> bell</c><00:13:06.720><c> to</c><00:13:06.880><c> be</c><00:13:06.959><c> notified</c><00:13:07.600><c> so</c><00:13:07.760><c> you</c>

00:13:07.910 --> 00:13:07.920 align:start position:0%
the little bell to be notified so you
 

00:13:07.920 --> 00:13:09.110 align:start position:0%
the little bell to be notified so you
never<00:13:08.240><c> miss</c><00:13:08.560><c> a</c><00:13:08.639><c> video</c>

00:13:09.110 --> 00:13:09.120 align:start position:0%
never miss a video
 

00:13:09.120 --> 00:13:10.870 align:start position:0%
never miss a video
thank<00:13:09.360><c> you</c><00:13:09.600><c> so</c><00:13:09.760><c> much</c><00:13:10.000><c> for</c><00:13:10.160><c> watching</c><00:13:10.720><c> and</c>

00:13:10.870 --> 00:13:10.880 align:start position:0%
thank you so much for watching and
 

00:13:10.880 --> 00:13:12.389 align:start position:0%
thank you so much for watching and
supporting<00:13:11.360><c> this</c><00:13:11.519><c> channel</c><00:13:11.920><c> it</c><00:13:12.000><c> really</c><00:13:12.240><c> does</c>

00:13:12.389 --> 00:13:12.399 align:start position:0%
supporting this channel it really does
 

00:13:12.399 --> 00:13:13.910 align:start position:0%
supporting this channel it really does
mean<00:13:12.639><c> the</c><00:13:12.720><c> absolute</c><00:13:13.120><c> world</c><00:13:13.360><c> to</c><00:13:13.519><c> me</c>

00:13:13.910 --> 00:13:13.920 align:start position:0%
mean the absolute world to me
 

00:13:13.920 --> 00:13:15.350 align:start position:0%
mean the absolute world to me
i<00:13:14.000><c> hope</c><00:13:14.160><c> you're</c><00:13:14.320><c> having</c><00:13:14.560><c> a</c><00:13:14.639><c> wonderful</c><00:13:15.120><c> day</c>

00:13:15.350 --> 00:13:15.360 align:start position:0%
i hope you're having a wonderful day
 

00:13:15.360 --> 00:13:16.949 align:start position:0%
i hope you're having a wonderful day
wherever<00:13:15.760><c> you</c><00:13:15.839><c> are</c><00:13:16.000><c> in</c><00:13:16.079><c> the</c><00:13:16.160><c> world</c><00:13:16.720><c> and</c><00:13:16.880><c> i</c>

00:13:16.949 --> 00:13:16.959 align:start position:0%
wherever you are in the world and i
 

00:13:16.959 --> 00:13:18.710 align:start position:0%
wherever you are in the world and i
can't<00:13:17.200><c> wait</c><00:13:17.360><c> to</c><00:13:17.519><c> see</c><00:13:17.680><c> you</c><00:13:17.839><c> here</c><00:13:18.000><c> again</c><00:13:18.399><c> next</c>

00:13:18.710 --> 00:13:18.720 align:start position:0%
can't wait to see you here again next
 

00:13:18.720 --> 00:13:19.430 align:start position:0%
can't wait to see you here again next
week

00:13:19.430 --> 00:13:19.440 align:start position:0%
week
 

00:13:19.440 --> 00:13:33.900 align:start position:0%
week
so<00:13:19.600><c> much</c><00:13:19.839><c> love</c><00:13:32.839><c> bye</c>

00:13:33.900 --> 00:13:33.910 align:start position:0%
so much love bye
 

00:13:33.910 --> 00:13:44.310 align:start position:0%
so much love bye
[Music]

00:13:44.310 --> 00:13:44.320 align:start position:0%
 
 

00:13:44.320 --> 00:13:46.399 align:start position:0%
 
you
"""
# %%
# transcript_data = {
#     "content": content,
#     "mach_content": mach_content,
#     "subtitle_file": "test.srt",
# }
# %%
# segments = chunk_transcript(transcript_data=transcript_data, mach_transcript=False)
# man_timestamps = [
#     segment["timestamp"].split("_")
#     for segment in segments
#     if not (segment["seg_content"] == "" or segment["seg_content"] == "WEBVTT\n\n")
# ]
# man_timestamps
# %%
# mach_segments = chunk_mach_transcript(
#     transcript_data=transcript_data, man_timestamps=man_timestamps, in_memory=True
# )
# mach_timestamps = [segment["timestamp"].split("_") for segment in mach_segments]
# mach_timestamps
# %%
# new_segments = []
# if mach_segments is None:
#     for segment in segments:
#         if segment["in_manifest"] is True:
#             segment["mach_seg_content"] = "None"
#             segment["seg_text"] = "None"
#             segment["mach_seg_text"] = "None"
#             segment["mach_timestamps"] = ""
#             segment["edit_dist"] = 0.0
#             new_segments.append(segment)
# else:
#     mach_segments = deque(mach_segments)
#     for segment in segments:
#         seg_text = get_seg_text(segment)
#         if seg_text != "":
#             if len(mach_segments) == 0:
#                 norm_seg_text = normalizer(seg_text)
#                 segment["seg_text"] = norm_seg_text
#                 segment["mach_seg_text"] = ""
#                 segment["mach_seg_content"] = ""
#                 segment["mach_timestamps"] = ""
#                 # edit_dist = Levenshtein.distance(norm_seg_text, "")
#                 edit_dist = jiwer.wer(norm_seg_text, "")
#                 segment["edit_dist"] = edit_dist
#                 new_segments.append(segment)
#             else:
#                 mach_segment = mach_segments.popleft()
#                 if mach_segment["seg_content"] == "WEBVTT\n\n":
#                     while mach_segment["seg_content"] == "WEBVTT\n\n":
#                         if len(mach_segments) == 0:
#                             mach_segment = None
#                             break
#                         else:
#                             mach_segment = mach_segments.popleft()
#                 mach_seg_text = get_mach_seg_text(mach_segment)
#                 norm_seg_text = normalizer(seg_text)
#                 norm_mach_seg_text = normalizer(mach_seg_text)
#                 segment["seg_text"] = norm_seg_text
#                 segment["mach_seg_text"] = norm_mach_seg_text
#                 # edit_dist = Levenshtein.distance(norm_seg_text, norm_mach_seg_text)
#                 edit_dist = jiwer.wer(norm_seg_text, norm_mach_seg_text)
#                 segment["mach_seg_content"] = mach_segment["seg_content"]
#                 segment["mach_timestamps"] = mach_segment["timestamp"]
#                 segment["edit_dist"] = edit_dist
#                 new_segments.append(segment)
#         elif seg_text == "":
#             segment["seg_text"] = ""
#             segment["mach_seg_text"] = ""
#             segment["mach_seg_content"] = ""
#             segment["mach_timestamps"] = ""
#             segment["edit_dist"] = 0.0
#             new_segments.append(segment)
# # %%
# import textwrap
# for segment in sorted(new_segments, key=lambda s: s["edit_dist"], reverse=True):
#     print(f"{segment["timestamp"]=}, {segment["mach_timestamps"]=}\n")
#     print(f"{segment["edit_dist"]=}\n")
#     print(f"seg_text={textwrap.fill(segment["seg_text"], width=80)}\n")
#     print(f"mach_seg_text={textwrap.fill(segment["mach_seg_text"], width=80)}\n")
# %%
import json

with open("/Users/huongn/Downloads/shard_00000000.jsonl (5)", "r") as f:
    data = [json.loads(line.strip()) for line in f]
for d in data:
    if "TpkuuUyEAC8" == d["id"]:
        break
# %%
segments = chunk_transcript(transcript_data=d, mach_transcript=False)
# %%
man_timestamps = [
    segment["timestamp"].split("_")
    for segment in segments
    if not (segment["seg_content"] == "" or segment["seg_content"] == "WEBVTT\n\n")
]
# %%
man_timestamps
# %%
mach_segments = chunk_mach_transcript(
    transcript_data=d, man_timestamps=man_timestamps, in_memory=True
)
# %%
mach_timestamps = [
    segment["timestamp"].split("_")
    for segment in mach_segments
]
# %%
mach_timestamps
# %%
