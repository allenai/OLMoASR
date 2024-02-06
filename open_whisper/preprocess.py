# break audio files into 30-second segments paired with subset of transcript that occurs within that time segment
# audio is resampled to 16000 hz, 80-channel log-magnitude mel spectrogram, computed with a 25ms window and 10ms hop size
# spectrogram is a visual representation of spectrum of frequencies of a signal as it varies with time
# mel spectrogram is a spectrogram that is generated using the Mel scale, a perceptual scale of pitches judged by listeners to be equal in distance from one another
# 80-channel mel spectrogram is a mel spectrogram represented by 80 mel bins (frequency channels)
# feature normalization: globally scale input to be between [-1, 1] with approximate mean 0 across pre-training dataset
import os
import subprocess
import pandas as pd
import multiprocessing
from tqdm import tqdm
from itertools import repeat
from typing import Union
import utils


def download_transcript(
    video_id: str, lang_code: str, output_dir: str, sub_format: str = "srt"
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

    if sub_format == "srt":
        command.extend(["--convert-subs", "srt"])

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return


def parallel_download_transcript(args) -> None:
    download_transcript(*args)


def download_audio(video_id: str, output_dir: str, ext: str = "m4a") -> None:
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


def parallel_download_audio(args) -> None:
    download_audio(*args)


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


def parallel_chunk_audio_transcript_text(args) -> None:
    chunk_audio_transcript_text(*args)


def chunk_audio_transcript(
    transcript_file: str, audio_file: str, transcript_ext: str = "srt"
) -> None:
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

    transcript, *_ = utils.TranscriptReader(transcript_file).read()

    # if transcript_ext == "vtt":
    #     transcript, *_ = read_vtt(transcript_file)
    # else:
    #     transcript, *_ = read_srt(transcript_file)

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
        init_diff = utils.calculate_difference(timestamps[a][0], timestamps[b][1])
        if init_diff < 30000:
            diff = init_diff
            b += 1
        else:
            t_output_file = f"{t_output_dir}/{timestamps[a][0]}_{timestamps[b - 1][1]}.{transcript_ext}"

            # write vtt file
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
            a = b

        if b == len(transcript) and diff < 30000:
            t_output_file = f"{t_output_dir}/{timestamps[a][0]}_{timestamps[b - 1][1]}.{transcript_ext}"

            # write vtt file
            utils.write_segment(
                timestamps[a:b],
                transcript,
                t_output_dir,
                transcript_ext,
            )

            utils.trim_audio(
                audio_file, timestamps[a][0], timestamps[b - 1][1], 0, 0, a_output_dir
            )

            break


def parallel_chunk_audio_transcript(args) -> None:
    chunk_audio_transcript(*args)


if __name__ == "__main__":
    # video_id = "eh77AUKedyM"
    # transcript_file = "data/transcripts/eh77AUKedyM/eh77AUKedyM.en-US.srt"
    # audio_file = "data/audio/eh77AUKedyM/eh77AUKedyM.m4a"
    # chunk_audio_transcript(transcript_file, audio_file)

    transcript_ext = "srt"
    audio_ext = "m4a"

    # reading in metadata
    df = pd.read_parquet("data/metadata/captions-0010.parquet")
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
                    parallel_download_audio,
                    zip(sample_id, repeat("data/audio"), repeat(audio_ext)),
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
                    zip(
                        sample_id,
                        sample_lang,
                        repeat("data/transcripts"),
                        repeat(transcript_ext),
                    ),
                ),
                total=len(sample),
            )
        )

    # transcript and audio file paths for reference when chunking
    transcript_file_paths = [
        f"data/transcripts/{sample_id[i]}/{sample_id[i]}.{sample_lang[i]}.{transcript_ext}"
        for i in range(len(sample_id))
    ]
    audio_file_paths = [f"data/audio/{id}/{id}.{audio_ext}" for id in sample_id]

    # for i in range(len(sample_id)):
    #     print(f"Processing {sample_id[i]}")
    #     chunk_audio_transcript(transcript_file_paths[i], audio_file_paths[i])

    # chunking audios and transcripts
    with multiprocessing.Pool() as pool:
        out = list(
            tqdm(
                pool.imap_unordered(
                    parallel_chunk_audio_transcript,
                    zip(
                        transcript_file_paths,
                        audio_file_paths,
                        repeat(transcript_ext),
                    ),
                ),
                total=len(sample),
            )
        )
