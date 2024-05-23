import os
import multiprocessing
from tqdm import tqdm
from open_whisper.utils import TranscriptReader, calculate_difference
from pydub import AudioSegment
from datetime import timedelta, datetime, time


def get_start_end(transcript_path: str, audio_path: str) -> None:
    """Get timestamps from audio and transcript files

    Args:
        transcript_path: Path to the transcript file
        audio_path: Path to the audio file
    """
    try:
        # Load transcript
        video_id = audio_path.split("/")[-1].split(".")[0]
        reader = TranscriptReader(transcript_path)
        _, t_start, t_end = reader.read()
        t_duration = calculate_difference(t_start, t_end) / 1000.0
        t_start = ":".join(t_start.split(".")[:-1])
        t_end = ":".join(t_end.split(".")[:-1])

        # Load audio
        audio = AudioSegment.from_file(audio_path)

        # Get duration in milliseconds
        duration_milliseconds = len(audio)
        duration_seconds = duration_milliseconds / 1000.0
        a_end = (
            (
                datetime.combine(datetime(2000, 1, 1), time(0, 0, 0, 0))
                + timedelta(seconds=duration_seconds)
            )
            .time()
            .strftime("%H:%M:%S")
        )

        with open("logs/data/preprocess/start_end.txt", "a") as f:
            f.write(
                f"{video_id}\t{t_start}\t{t_end}\t{t_duration}\t{a_end}\t{duration_seconds}\n"
            )
    except Exception as e:
        with open("logs/data/preprocess/fail_retrieve_start_end.txt", "a") as f:
            f.write(f"{transcript_path}\t{audio_path}\t{e}\n")

    return None


def parallel_get_start_end(args: tuple) -> None:
    """Parallel function to get timestamps from audio and transcript files

    Args:
        args: Tuple containing the transcript path and audio path
    """
    get_start_end(*args)


def main():
    with open("logs/data/download/transcript_paths.txt", "r") as f:
        transcript_file_paths = f.read().splitlines()

    with open("logs/data/download/audio_paths.txt", "r") as f:
        audio_file_paths = f.read().splitlines()

    with multiprocessing.Pool(multiprocessing.cpu_count() * 7) as pool:
        out = list(
            tqdm(
                pool.imap_unordered(
                    parallel_get_start_end,
                    zip(
                        transcript_file_paths,
                        audio_file_paths,
                    ),
                ),
                total=len(transcript_file_paths),
            )
        )


if __name__ == "__main__":
    main()
