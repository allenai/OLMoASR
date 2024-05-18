from datetime import datetime, timedelta
import re
import subprocess
from typing import Dict, Union, Tuple, List, Optional
import pysrt
import webvtt
import jiwer
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
from whisper import utils


CHARS_TO_REMOVE = ["&nbsp;", r"\\h", r"\\h\\h"]


def remove_after_endoftext(text: str) -> str:
    """Removes everything after the first instance of "<|endoftext|>" in a string.

    Args:
        text: The string to process

    Returns:
        The string with everything after the first "<|endoftext|>" removed.
    """
    endoftext_index = text.find("<|endoftext|>")
    if endoftext_index != -1:
        return text[: endoftext_index + len("<|endoftext|>")]
    else:
        return text


def convert_to_milliseconds(timestamp: str) -> int:
    """Convert a timestamp in the format HH:MM:SS.mmm to milliseconds

    Args:
        timestamp: Timestamp in the format HH:MM:SS.mmm

    Returns:
        Timestamp in milliseconds
    """
    h, m, s, ms = map(float, timestamp.replace(".", ":").split(":"))
    return int(h * 3600000 + m * 60000 + s * 1000 + ms)


def calculate_difference(timestamp1: str, timestamp2: str) -> int:
    """Calculate the difference between two timestamps in milliseconds

    Args:
        timestamp1: Timestamp in the format HH:MM:SS.mmm
        timestamp2: Timestamp in the format HH:MM:SS.mmm

    Returns:
        Difference between the two timestamps in milliseconds
    """
    time1 = convert_to_milliseconds(timestamp1)
    time2 = convert_to_milliseconds(timestamp2)
    if time2 < time1:
        raise ValueError(
            "Second timestamp is less than the first timestamp. Needs to be greater than the first timestamp."
        )
    return time2 - time1


def adjust_timestamp(timestamp: str, milliseconds: int) -> str:
    """Adjust a timestamp by a specified number of milliseconds

    Args:
        timestamp: Timestamp in the format HH:MM:SS.mmm
        milliseconds: Number of milliseconds to add or subtract from the timestamp

    Returns:
        Adjusted timestamp in the format HH:MM:SS.mmm
    """
    # Convert the HH:MM:SS.mmm format to a datetime object
    original_time = datetime.strptime(timestamp, "%H:%M:%S.%f")

    # Adjust the time by the specified number of seconds
    # Use timedelta(milliseconds=milliseconds) to add or timedelta(milliseconds=-milliseconds) to subtract
    adjusted_time = original_time + timedelta(milliseconds=milliseconds)

    # Convert back to the HH:MM:SS.mmm string format
    return adjusted_time.strftime("%H:%M:%S.%f")[
        :-3
    ]  # Truncate microseconds to milliseconds


def trim_audio(
    audio_file: str,
    start: str,
    end: Optional[str],
    output_dir: str,
    start_window: int = 0,
    end_window: int = 0,
) -> None:
    """Trim an audio file to a specified start and end timestamp

    Trims the audio file to the specified start and end timestamps and saves the trimmed audio file to the output directory.
    If start_window and end_window are specified, the audio file will be trimmed by the specified number of seconds before/after the start timestamp and before/after the end timestamp.
    For before, start_window and end_window should be negative values. For after, they should be positive values.

    Args:
        audio_file: Path to the audio file
        start: Start timestamp in the format HH:MM:SS.mmm
        end: End timestamp in the format HH:MM:SS.mmm
        start_window: Number of seconds to add to the start timestamp
        end_window: Number of seconds to add from the end timestamp
        output_dir: Directory to save the trimmed audio file
    """
    adjusted_start = adjust_timestamp(timestamp=start, milliseconds=start_window * 1000)
    adjusted_end = adjust_timestamp(timestamp=end, milliseconds=end_window * 1000)

    command = [
        "ffmpeg",
        "-i",
        audio_file,
        "-ss",
        adjusted_start,
    ]

    if end is not None:
        command.append(["-to", adjusted_end])
        command.append(
            ["-c", "copy", f"{output_dir}/{start}_{end}.{audio_file.split('.')[-1]}"]
        )
    else:
        command.append(
            ["-c", "copy", f"{output_dir}/{start}.{audio_file.split('.')[-1]}"]
        )

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class TranscriptReader:
    """A class to read in a WebVTT or SRT transcript file and extract the transcript

    Attributes:
        file_path: Path to the transcript file
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.ext = file_path.split(".")[-1]

    def read_vtt(self, file_path: str) -> Union[None, Tuple[Dict, str, str]]:
        """Read a WebVTT file

        Args:
            file_path: Path to the WebVTT file

        Returns:
            A tuple containing the transcript, start timestamp, and end timestamp or None if the file is empty
        """
        transcript = {}
        captions = webvtt.read(file_path)

        if len(captions) == 0:
            return transcript, "", ""

        transcript_start = captions[0].start
        transcript_end = captions[-1].end
        for caption in captions:
            start = caption.start
            end = caption.end
            text = caption.text
            transcript[(start, end)] = text

        return transcript, transcript_start, transcript_end

    def read_srt(self, file_path: str) -> Union[None, Tuple[Dict, str, str]]:
        """Read an SRT file

        Args:
            file_path: Path to the SRT file

        Returns:
            A tuple containing the transcript, start timestamp, and end timestamp or None if the file is empty
        """
        transcript = {}
        subs = pysrt.open(file_path)
        if len(subs) == 0:
            return transcript, "", ""

        transcript_start = f"{subs[0].start.hours:02}:{subs[0].start.minutes:02}:{subs[0].start.seconds:02}.{subs[0].start.milliseconds:03}"
        transcript_end = f"{subs[-1].end.hours:02}:{subs[-1].end.minutes:02}:{subs[-1].end.seconds:02}.{subs[-1].end.milliseconds:03}"
        for sub in subs:
            start = f"{sub.start.hours:02}:{sub.start.minutes:02}:{sub.start.seconds:02}.{sub.start.milliseconds:03}"
            end = f"{sub.end.hours:02}:{sub.end.minutes:02}:{sub.end.seconds:02}.{sub.end.milliseconds:03}"
            text = sub.text
            transcript[(start, end)] = text

        return transcript, transcript_start, transcript_end

    def read(self) -> Union[None, Tuple[Dict, str, str]]:
        """Read the transcript file

        Returns:
            A tuple containing the transcript, start timestamp, and end timestamp or None if the file is empty
        """
        if self.ext == "vtt":
            return self.read_vtt(self.file_path)
        elif self.ext == "srt":
            return self.read_srt(self.file_path)
        else:
            raise ValueError("Unsupported file type")

    def extract_text(self, transcript: Dict) -> Optional[str]:
        """Extract the text from the transcript

        Args:
            transcript: Transcript as a dictionary

        Returns:
            The extracted text
        """
        transcript_text = ""
        for _, text in transcript.items():
            transcript_text += text.strip() + " "
        return transcript_text.strip()


def write_segment(
    timestamps: List, transcript: Optional[Dict], output_dir: str, ext: str
) -> None:
    """Write a segment of the transcript to a file

    Args:
        timestamps: List of timestamps
        transcript: Transcript as a dictionary
        output_dir: Directory to save the transcript file
        ext: File extension
    """
    with open(f"{output_dir}/{timestamps[0][0]}_{timestamps[-1][1]}.{ext}", "w") as f:
        if transcript == None:
            f.write("")
        else:
            if ext == "vtt":
                f.write("WEBVTT\n\n")

            for i in range(len(timestamps)):
                start = adjust_timestamp(
                    timestamp=timestamps[i][0],
                    milliseconds=-convert_to_milliseconds(timestamps[0][0]),
                )
                end = adjust_timestamp(
                    timestamp=timestamps[i][1],
                    milliseconds=-convert_to_milliseconds(timestamps[0][0]),
                )

                if ext == "srt":
                    start = start.replace(".", ",")
                    end = end.replace(".", ",")
                    f.write(f"{i + 1}\n")

                f.write(
                    f"{start} --> {end}\n{transcript[(timestamps[i][0], timestamps[i][1])]}\n\n"
                )


def calculate_wer(pair: Tuple[str, str]) -> float:
    """Calculate the Word Error Rate (WER) between two strings

    Args:
        pair: A tuple containing the truth and predicted strings

    Returns:
        The Word Error Rate (WER) between the two strings
    """
    # truth, predicted
    if pair[0] == "":
        return 0.0
    else:
        return jiwer.wer(pair[0], pair[1]) * 100.0


def clean_transcript(file_path) -> Union[None, bool]:
    """Remove unnecessary characters from the transcript file

    Args:
        file_path: Path to the transcript file
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    if content.strip() == "":
        return None

    # Replace &nbsp; with a space or an empty string
    regex_pattern = "|".join(CHARS_TO_REMOVE)
    modified_content = re.sub(regex_pattern, "", content)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(modified_content)

    return True
