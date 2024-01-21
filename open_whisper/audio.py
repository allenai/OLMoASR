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
        "bestaudio[ext=m4a]",
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


def read_vtt(file_path: str) -> dict:
    transcript = {}
    captions = webvtt.read(file_path)
    for caption in captions:
        start = caption.start
        end = caption.end
        text = caption.text
        transcript[(start, end)] = text

    return transcript


def chunk_audio(audio_file: str, output_dir: str) -> None:
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
        f"{output_dir}/%(id)s/%(id)s_%03d.m4a",
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def chunk_transcript(transcript_file: str, output_dir: str) -> None:
    transcript = read_vtt(transcript_file)
    a = 0
    b = 0
    timestamps = transcript.keys()
    start = timestamps[a][0]
    end = timestamps[b][1]
    start_ms = convert_to_milliseconds(start)
    end_ms = convert_to_milliseconds(end)
    init_diff = calculate_difference(start_ms, end_ms)
    text = transcript[(start, end)]
    added_silence = False

    while a < len(transcript):
        if init_diff < 30000:
            b += 1
            if convert_to_milliseconds(timestamps[b][0]) > convert_to_milliseconds(
                timestamps[a][1]
            ):
                init_diff += calculate_difference(
                    convert_to_milliseconds(timestamps[b][0]),
                    convert_to_milliseconds(timestamps[a][1]),
                )
                added_silence = True
                continue

            added_silence = False
            start = timestamps[b][0]
            end = timestamps[b][1]
            start_ms = convert_to_milliseconds(start)
            end_ms = convert_to_milliseconds(end)

            diff = calculate_difference(start_ms, end_ms)
            init_diff += diff

            text += transcript[(start, end)]
        else:
            # don't know what to do here yet so slay
            if init_diff >= 31000:
                if added_silence:
                    a = b
                else:
                    # depending on how many words are in that time segment, we may need to remove some words
                    text = text[: -len(transcript[(start, end)])]
            else:
                b += 1
                a = b

            transcript_file = open(f"{output_dir}/{start}_{end}.txt", "w")
            transcript_file.write(text)
            transcript_file.close()
            start = timestamps[a][0]
            end = timestamps[b][1]
            start_ms = convert_to_milliseconds(start)
            end_ms = convert_to_milliseconds(end)
            init_diff = calculate_difference(start_ms, end_ms)
