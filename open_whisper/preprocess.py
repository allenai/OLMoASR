# break audio files into 30-second segments paired with subset of transcript that occurs within that time segment
# audio is resampled to 16000 hz, 80-channel log-magnitude mel spectrogram, computed with a 25ms window and 10ms hop size
# spectrogram is a visual representation of spectrum of frequencies of a signal as it varies with time
# mel spectrogram is a spectrogram that is generated using the Mel scale, a perceptual scale of pitches judged by listeners to be equal in distance from one another
# 80-channel mel spectrogram is a mel spectrogram represented by 80 mel bins (frequency channels)
# feature normalization: globally scale input to be between [-1, 1] with approximate mean 0 across pre-training dataset
import os
import subprocess
import webvtt
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from itertools import repeat
from typing import Dict, Tuple, Union, List
from datetime import datetime, timedelta


def download_transcript(
    video_id: str, lang_code: str, output_dir: str, sub_format: str
) -> None:
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

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return


def parallel_download_transcript(args) -> None:
    download_transcript(*args)


def download_audio(video_id: str, output_dir: str, ext: str = "m4a") -> None:
    command = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "-f",
        f"bestaudio[ext={ext}][asr=44][acodec=mp4a]",
        "--audio-quality",
        "0",
        "-o",
        f"{output_dir}/%(id)s/%(id)s.%(ext)s",
    ]

    if ext == "wav":
        command.extend(["--extract-audio", "--audio-format", "wav"])

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parallel_download_audio(args) -> None:
    download_audio(*args)


def convert_to_milliseconds(timestamp: str) -> int:
    h, m, s, ms = map(float, timestamp.replace(".", ":").split(":"))
    return int(h * 3600000 + m * 60000 + s * 1000 + ms)


def calculate_difference(timestamp1: str, timestamp2: str) -> int:
    time1 = convert_to_milliseconds(timestamp1)
    time2 = convert_to_milliseconds(timestamp2)
    return abs(time2 - time1)


def adjust_timestamp(timestamp: str, milliseconds: int) -> str:
    # Convert the HH:MM:SS.mmm format to a datetime object
    original_time = datetime.strptime(timestamp, "%H:%M:%S.%f")

    # Adjust the time by the specified number of seconds
    # Use timedelta(milliseconds=milliseconds) to add or timedelta(milliseconds=-milliseconds) to subtract
    adjusted_time = original_time + timedelta(milliseconds=milliseconds)

    # Convert back to the HH:MM:SS.mmm string format
    return adjusted_time.strftime("%H:%M:%S.%f")[
        :-3
    ]  # Truncate microseconds to milliseconds


def clean_transcript(file_path) -> Union[None, bool]:
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    if content.strip() == "":
        return None

    # Replace &nbsp; with a space or an empty string
    modified_content = content.replace("&nbsp;", " ")

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(modified_content)

    return True


def read_vtt(file_path: str) -> Union[None, Tuple[Dict, str, str]]:
    transcript = {}
    captions = webvtt.read(file_path)

    if captions == []:
        return transcript, "", ""

    transcript_start = captions[0].start
    transcript_end = captions[-1].end
    for caption in captions:
        start = caption.start
        end = caption.end
        text = caption.text
        transcript[(start, end)] = text

    return transcript, transcript_start, transcript_end


def trim_audio(
    audio_file: str,
    start: str,
    end: str,
    start_window: int,
    end_window: int,
    output_dir: str,
) -> None:
    adjusted_start = adjust_timestamp(timestamp=start, milliseconds=start_window * 1000)
    adjusted_end = adjust_timestamp(timestamp=end, milliseconds=end_window * 1000)

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


def write_vtt_segment(timestamps: List, transcript: Dict, output_dir: str) -> None:
    with open(f"{output_dir}/{timestamps[0][0]}_{timestamps[-1][1]}.vtt", "w") as f:
        f.write("WEBVTT\n")
        for i in range(len(timestamps)):
            start = adjust_timestamp(
                timestamp=timestamps[i][0],
                milliseconds=-convert_to_milliseconds(timestamps[0][0]),
            )
            end = adjust_timestamp(
                timestamp=timestamps[i][1],
                milliseconds=-convert_to_milliseconds(timestamps[0][0]),
            )
            f.write(
                f"{start} --> {end}\n{transcript[(timestamps[i][0], timestamps[i][1])]}\n\n"
            )


def chunk_audio_transcript_text(transcript_file: str, audio_file: str):
    # if transcript or audio files doesn't exist
    if not os.path.exists(transcript_file):
        with open(f"logs/failed_download_t.txt", "a") as f:
            f.write(f"{transcript_file}\n")
        if not os.path.exists(audio_file):
            with open(f"logs/failed_download_a.txt", "a") as f:
                f.write(f"{audio_file}\n")

        return None

    t_output_dir = "/".join(transcript_file.split("/")[:3]) + "/segments"
    a_output_dir = "/".join(audio_file.split("/")[:3]) + "/segments"
    os.makedirs(t_output_dir, exist_ok=True)
    os.makedirs(a_output_dir, exist_ok=True)

    cleaned_transcript = clean_transcript(transcript_file)
    if cleaned_transcript is None:
        with open(f"logs/empty_transcript.txt", "a") as f:
            f.write(f"{transcript_file}\n")
        return None

    transcript, *_ = read_vtt(transcript_file)

    # if transcript file is empty
    if transcript == {}:
        with open(f"logs/empty_transcript.txt", "a") as f:
            f.write(f"{transcript_file}\n")
        return None

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
            if text != "":
                if text[-1] != " ":
                    text += " "

            text += transcript[(timestamps[b][0], timestamps[b][1])]
            b += 1
        else:
            t_output_file = (
                f"{t_output_dir}/{timestamps[a][0]}_{timestamps[b - 1][1]}.txt"
            )
            transcript_file = open(t_output_file, "w")
            transcript_file.write(text)
            transcript_file.close()

            trim_audio(
                audio_file,
                timestamps[a][0],
                timestamps[b - 1][1],
                0,
                0,
                a_output_dir,
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


def chunk_audio_transcript(transcript_file: str, audio_file: str) -> None:
    # if transcript or audio files doesn't exist
    if not os.path.exists(transcript_file):
        with open(f"logs/failed_download_t.txt", "a") as f:
            f.write(f"{transcript_file}\n")
        if not os.path.exists(audio_file):
            with open(f"logs/failed_download_a.txt", "a") as f:
                f.write(f"{audio_file}\n")

        return None

    t_output_dir = "/".join(transcript_file.split("/")[:3]) + "/segments"
    a_output_dir = "/".join(audio_file.split("/")[:3]) + "/segments"
    os.makedirs(t_output_dir, exist_ok=True)
    os.makedirs(a_output_dir, exist_ok=True)

    cleaned_transcript = clean_transcript(transcript_file)
    if cleaned_transcript is None:
        with open(f"logs/empty_transcript.txt", "a") as f:
            f.write(f"{transcript_file}\n")
        return None

    transcript, *_ = read_vtt(transcript_file)

    # if transcript file is empty
    if transcript == {}:
        with open(f"logs/empty_transcript.txt", "a") as f:
            f.write(f"{transcript_file}\n")
        return None

    a = 0
    b = 0

    timestamps = list(transcript.keys())
    diff = 0
    init_diff = 0

    while a < len(transcript) + 1:
        init_diff = calculate_difference(timestamps[a][0], timestamps[b][1])
        if init_diff < 30000:
            diff = init_diff
            b += 1
        else:
            t_output_file = (
                f"{t_output_dir}/{timestamps[a][0]}_{timestamps[b - 1][1]}.vtt"
            )

            # write vtt file
            write_vtt_segment(
                timestamps[a:b],
                transcript,
                t_output_dir,
            )

            trim_audio(
                audio_file,
                timestamps[a][0],
                timestamps[b - 1][1],
                0,
                0,
                a_output_dir,
            )

            init_diff = 0
            diff = 0
            a = b

        if b == len(transcript) and diff < 30000:
            t_output_file = (
                f"{t_output_dir}/{timestamps[a][0]}_{timestamps[b - 1][1]}.vtt"
            )

            # write vtt file
            write_vtt_segment(
                timestamps[a:b],
                transcript,
                t_output_dir,
            )

            trim_audio(
                audio_file, timestamps[a][0], timestamps[b - 1][1], 0, 0, a_output_dir
            )

            break


def parallel_chunk_audio_transcript_text(args) -> None:
    chunk_audio_transcript_text(*args)


if __name__ == "__main__":
    video_id = "eh77AUKedyM"
    transcript_file = "data/transcripts/eh77AUKedyM/eh77AUKedyM.en-US.vtt"
    audio_file = "data/audio/eh77AUKedyM/eh77AUKedyM.wav"
    chunk_audio_transcript(transcript_file, audio_file)

    # reading in metadata
    df = pd.read_parquet("data/metadata/data.parquet")
    # only getting english data (that's less than 5 minutes long)
    en_df = df[
        (df["manual_caption_languages"].str.contains("en"))
        & (df["automatic_caption_orig_language"].str.contains("en"))
    ]

    # randomly sampling 36 videos
    # rng = np.random.default_rng(42)
    # sample = rng.choice(en_df[["id", "manual_caption_languages"]], 50, replace=False)

    sample = en_df[en_df["categories"] == "Education"][
        ["id", "manual_caption_languages"]
    ].to_numpy()[:10]
    # ensuring that language codes are english only
    for i, (id, langs) in enumerate(sample):
        if "," in langs:
            for lang in langs.split(","):
                if "en" in lang:
                    sample[i][1] = lang
                    break

    # breaking up the sample list (2 columns) into 2 lists
    sample_id, sample_lang = [row[0] for row in sample], [row[1] for row in sample]

    # downloading audio
    with multiprocessing.Pool() as pool:
        out = list(
            tqdm(
                pool.imap_unordered(
                    parallel_download_audio, zip(sample_id, repeat("data/audio"))
                ),
                total=len(sample),
            )
        )

    # downloading transcripts
    with multiprocessing.Pool() as pool:
        out = list(
            tqdm(
                pool.imap_unordered(
                    parallel_download_transcript,
                    zip(sample_id, sample_lang, repeat("data/transcripts")),
                ),
                total=len(sample),
            )
        )

    # transcript and audio file paths for reference when chunking
    transcript_file_paths = [
        f"data/transcripts/{sample_id[i]}/{sample_id[i]}.{sample_lang[i]}.vtt"
        for i in range(len(sample_id))
    ]
    audio_file_paths = [f"data/audio/{id}/{id}.wav" for id in sample_id]

    # for i in range(len(sample_id)):
    #     print(f"Processing {sample_id[i]}")
    #     chunk_audio_transcript(transcript_file_paths[i], audio_file_paths[i])

    # chunking audios and transcripts
    with multiprocessing.Pool() as pool:
        out = list(
            tqdm(
                pool.imap_unordered(
                    parallel_chunk_audio_transcript_text,
                    zip(
                        transcript_file_paths,
                        audio_file_paths,
                    ),
                ),
                total=len(sample),
            )
        )
