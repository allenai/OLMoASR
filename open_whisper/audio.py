# break audio files into 30-second segments paired with subset of transcript that occurs within that time segment
# audio is resampled to 16000 hz, 80-channel log-magnitude mel spectrogram, computed with a 25ms window and 10ms hop size
# spectrogram is a visual representation of spectrum of frequencies of a signal as it varies with time
# mel spectrogram is a spectrogram that is generated using the Mel scale, a perceptual scale of pitches judged by listeners to be equal in distance from one another
# 80-channel mel spectrogram is a mel spectrogram represented by 80 mel bins (frequency channels)
# feature normalization: globally scale input to be between [-1, 1] with approximate mean 0 across pre-training dataset
import yt_dlp
import os
import subprocess
import webvtt
import numpy as np
from typing import Dict, Tuple
from datetime import datetime, timedelta


def download_transcript(video_id: str, lang_code: str, output_dir: str) -> None:
    if lang_code == "unknown":
        lang_code = "en"

    command = [
        "yt-dlp",
        "--write-subs",
        "--no-write-auto-subs",
        "--skip-download",
        "--sub-format",
        "vtt",
        "--sub-langs",
        f"{lang_code},-live_chat",
        f"https://www.youtube.com/watch?v={video_id}",
        "-o",
        f"{output_dir}/%(id)s/%(id)s.%(ext)s",
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def download_audio(video_id: str, output_dir: str) -> None:
    command = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "-f",
        "bestaudio",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        f"{output_dir}/%(id)s/%(id)s.%(ext)s",
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def convert_to_milliseconds(timestamp: str) -> int:
    h, m, s, ms = map(float, timestamp.replace(".", ":").split(":"))
    return int(h * 3600000 + m * 60000 + s * 1000 + ms)


def calculate_difference(timestamp1: str, timestamp2: str) -> int:
    time1 = convert_to_milliseconds(timestamp1)
    time2 = convert_to_milliseconds(timestamp2)
    return abs(time2 - time1)


def adjust_timestamp(timestamp: str, seconds: int) -> str:
    # Convert the HH:MM:SS.mmm format to a datetime object
    original_time = datetime.strptime(timestamp, "%H:%M:%S.%f")

    # Adjust the time by the specified number of seconds
    # Use timedelta(seconds=seconds) to add or timedelta(seconds=-seconds) to subtract
    adjusted_time = original_time + timedelta(seconds=seconds)

    # Convert back to the HH:MM:SS.mmm string format
    return adjusted_time.strftime("%H:%M:%S.%f")[
        :-3
    ]  # Truncate microseconds to milliseconds


def read_vtt(file_path: str) -> Tuple[Dict, str, str]:
    transcript = {}
    captions = webvtt.read(file_path)
    transcript_start = captions[0].start
    transcript_end = captions[-1].end
    for caption in captions:
        start = caption.start
        end = caption.end
        text = caption.text
        transcript[(start, end)] = text

    return transcript, transcript_start, transcript_end


def trim_audio(
    audio_file: str, start: str, end: str, start_window: int, end_window: int, output_dir: str
) -> None:
    adjusted_start = adjust_timestamp(start, start_window)
    adjusted_end = adjust_timestamp(end, end_window)

    command = [
        "ffmpeg",
        "-i",
        audio_file,
        "-ss",
        adjusted_start,
        "-to",
        adjusted_end,
        "-c",
        "copy",
        f"{output_dir}/{start}_{end}.{audio_file.split('.')[-1]}",
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def chunk_audio(audio_file: str, output_dir: str, transcript_start: str) -> None:
    output_dir = output_dir + "/segments"
    os.makedirs(output_dir, exist_ok=True)

    command = [
        "ffmpeg",
        "-i",
        audio_file,
        "-f",
        "segment",
        "-segment_time",
        "30",
        "-c",
        "copy",
        f"{output_dir}/.{audio_file.split('.')[-1]}",
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for file in sorted(os.listdir(output_dir), key=lambda x: int(x.split(".")[0])):
        new_time = adjust_timestamp(transcript_start, 30)
        os.rename(
            os.path.join(output_dir, file),
            os.path.join(
                output_dir, f"{transcript_start}_{new_time}.{file.split('.')[-1]}"
            ),
        )
        transcript_start = new_time


def chunk_audio_transcript(
    transcript_file: str, audio_file: str, t_output_dir: str, a_output_dir: str
):
    t_output_dir = t_output_dir + "/segments"
    a_output_dir = a_output_dir + "/segments"
    os.makedirs(t_output_dir, exist_ok=True)
    os.makedirs(a_output_dir, exist_ok=True)

    transcript, *_ = read_vtt(transcript_file)
    a = 0
    b = 0

    timestamps = list(transcript.keys())
    diff = 0
    init_diff = 0
    text = ""

    while a < len(transcript) + 1:
        init_diff = calculate_difference(timestamps[a][0], timestamps[b][1])
        if init_diff < 30000:
            diff = init_diff
            text += transcript[(timestamps[b][0], timestamps[b][1])]
            b += 1
        else:
            t_output_file = (
                f"{t_output_dir}/{timestamps[a][0]}_{timestamps[b - 1][1]}.txt"
            )
            transcript_file = open(t_output_file, "w")
            transcript_file.write(text)
            transcript_file.close()

            if a == 0:
                start_window = -2
            else:
                start_window = 0

            end_window = 0

            trim_audio(
                audio_file, timestamps[a][0], timestamps[b - 1][1], start_window, end_window, a_output_dir
            )
            text = ""
            init_diff = 0
            diff = 0
            a = b

        if b == len(transcript) and diff < 30000:
            t_output_file = (
                f"{t_output_dir}/{timestamps[a][0]}_{timestamps[b - 1][1]}.txt"
            )
            transcript_file = open(t_output_file, "w")
            transcript_file.write(text)
            transcript_file.close()

            trim_audio(
                audio_file, timestamps[a][0], timestamps[b - 1][1], 0, 0, a_output_dir
            )

            break


# def chunk_transcript(transcript_file: str, output_dir: str) -> None:
#     output_dir = output_dir + "/segments"
#     os.makedirs(output_dir, exist_ok=True)

#     transcript, *_ = read_vtt(transcript_file)
#     a = 0
#     b = 0
#     timestamps = list(transcript.keys())
#     start = timestamps[a][0]
#     # end = timestamps[b][1]
#     # text = transcript[(start, end)]
#     init_diff = 0
#     text = ""
#     added_silence = False
#     remain_text = ""

#     while a < len(transcript) + 1:
#         if init_diff < 30000:
#             if init_diff == 0 and convert_to_milliseconds(
#                 adjust_timestamp(start, 30)
#             ) < convert_to_milliseconds(timestamps[a][0]):
#                 init_diff = 30000
#             elif a == len(transcript) - 1:
#                 text += transcript[(timestamps[a][0], timestamps[a][1])]
#                 init_diff = 30000
#                 a += 1
#             else:
#                 init_diff = calculate_difference(start, timestamps[b][1])
#                 text += transcript[(timestamps[a][0], timestamps[a][1])]
#                 b += 1
#                 if convert_to_milliseconds(timestamps[b][0]) > convert_to_milliseconds(
#                     timestamps[a][1]
#                 ):
#                     init_diff += calculate_difference(
#                         timestamps[b][0],
#                         timestamps[a][1],
#                     )
#                     added_silence = True
#                 a += 1
#         else:
#             new_start = adjust_timestamp(start, 30)
#             output_file = f"{output_dir}/{start}_{new_start}.txt"
#             remain_text = ""
#             if init_diff >= 31000:
#                 if not added_silence:
#                     if (init_diff - 30000) // 1000 > len(
#                         transcript[(timestamps[a - 1][0], timestamps[a - 1][1])].split(
#                             " "
#                         )
#                     ):
#                         keep_len = int(
#                             np.ceil(
#                                 ((init_diff - 30000) // 1000)
#                                 / len(
#                                     transcript[
#                                         (timestamps[a - 1][0], timestamps[a - 1][1])
#                                     ].split(" ")
#                                 )
#                                 * 1.0
#                             )
#                         )
#                         tokens = text.split(" ")
#                         tokens = tokens[
#                             : -(
#                                 len(
#                                     transcript[
#                                         (timestamps[a - 1][0], timestamps[a - 1][1])
#                                     ].split(" ")
#                                 )
#                                 - keep_len
#                             )
#                         ]
#                         text = " ".join(tokens)
#                         remain_text = transcript[
#                             (timestamps[a - 1][0], timestamps[a - 1][1])
#                         ][keep_len:]
#                     else:  # < or ==
#                         tokens = text.split(" ")
#                         text = " ".join(
#                             tokens[
#                                 : -len(
#                                     transcript[
#                                         (timestamps[a - 1][0], timestamps[a - 1][1])
#                                     ].split(" ")
#                                 )
#                                 + 1
#                             ]
#                         )
#                         remain_text = " ".join(
#                             transcript[
#                                 (timestamps[a - 1][0], timestamps[a - 1][1])
#                             ].split(" ")[1:]
#                         )

#                     a -= 1
#                     b -= 1

#             transcript_file = open(output_file, "w")
#             transcript_file.write(text)
#             transcript_file.close()

#             if a == len(transcript):
#                 break

#             init_diff = 0
#             text = remain_text
#             start = new_start


if __name__ == "__main__":
    video_id = "-t-J098gF10"
    transcript_file = "data/transcripts/-t-J098gF10/-t-J098gF10.en-uYU-mmqFLq8.vtt"
    audio_file = "data/audio/-t-J098gF10/-t-J098gF10.wav"
    t_output_dir = "data/transcripts/-t-J098gF10"
    a_output_dir = "data/audio/-t-J098gF10"
    # chunk_transcript(transcript_file, output_dir)
    chunk_audio_transcript(transcript_file, audio_file, t_output_dir, a_output_dir)
