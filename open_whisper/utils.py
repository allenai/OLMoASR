from datetime import datetime, timedelta
import re
import os
import shutil
import subprocess
from typing import Dict, Union, Tuple, List, Optional, Literal
import pysrt
import webvtt
from pydub import AudioSegment
import jiwer
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
from whisper import utils
from whisper.tokenizer import get_tokenizer


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
) -> str:
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

    Returns:
        Path to the trimmed audio file
    """
    if start_window > 0:
        adjusted_start = adjust_timestamp(
            timestamp=start, milliseconds=start_window * 1000
        )
    else:
        adjusted_start = start

    if end_window > 0:
        adjusted_end = adjust_timestamp(timestamp=end, milliseconds=end_window * 1000)
    else:
        adjusted_end = end

    output_file = f"{output_dir}/{start}_{end}.{audio_file.split('.')[-1]}"

    command = [
        "ffmpeg",
        "-i",
        audio_file,
        "-ss",
        adjusted_start,
    ]

    if end is not None:
        command.extend(
            [
                "-to",
                adjusted_end,
                "-c",
                "copy",
                output_file,
            ]
        )
    else:
        command.extend(["-c", "copy", output_file])

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_file


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
) -> str:
    """Write a segment of the transcript to a file

    Args:
        timestamps: List of timestamps
        transcript: Transcript as a dictionary
        output_dir: Directory to save the transcript file
        ext: File extension

    Returns:
        Path to the output transcript file
    """
    output_file = f"{output_dir}/{timestamps[0][0]}_{timestamps[-1][1]}.{ext}"
    with open(output_file, "w") as f:
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

    return output_file


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


def over_ctx_len(timestamps: List, transcript: Optional[Dict]) -> bool:
    """Check if transcript text exceeds model context length

    Check if the total number of tokens in the transcript text exceeds the model context length

    Args:
        timestamps: List of timestamps
        transcript: Transcript as a dictionary

    Returns:
        True if the transcript text exceeds the model context length, False otherwise
    """
    text_lines = [transcript[timestamps[i]].strip() for i in range(len(timestamps))]
    text = " ".join(text_lines)

    tokenizer = get_tokenizer(multilingual=False)

    text_tokens = tokenizer.encode(text)
    text_tokens = list(tokenizer.sot_sequence_including_notimestamps) + text_tokens
    text_tokens.append(tokenizer.eot)

    if len(text_tokens) > 448:
        return True
    else:
        return False


def too_short_audio(file_path: str) -> bool:
    timestamps = file_path.split(".m4a")[0].split("_")
    if calculate_difference(timestamps[0], timestamps[1]) < 15:
        return True
    return False


def corrupted_audio(file_path: str) -> bool:
    """Check if audio can't be loaded

    Check if audio can't be loaded because 1. too short 2. corrupted

    Args:
        file_path: Path to the audio file

    Returns:
        True if the audio can't be loaded, False otherwise
    """
    try:
        load_audio(file_path)
    except:
        return True
    return False


def check_audio(
    audio_file: str,
    transcript_file: str,
    video_id_dir: str,
) -> Optional[Literal["too short", "corrupted"]]:
    """Check if audio is corrupted and log it

    Args:
        audio_file: Path to the audio file
        transcript_file: Path to the transcript file
        video_id_dir: Directory of original audio and transcript files
    """
    if corrupted_audio(file_path=audio_file):
        shutil.rmtree(video_id_dir)
        return "corrupted"
        
    if too_short_audio(file_path=audio_file):
        os.remove(audio_file)
        os.remove(transcript_file)
        return "too short"
    
    return None
