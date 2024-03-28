import numpy as np
from datetime import datetime, timedelta
import subprocess
from typing import Dict, Union, Tuple, List, Optional
import pysrt
import webvtt
import jiwer
import zlib
from .normalizers import BasicTextNormalizer, EnglishTextNormalizer


def exact_div(x, y):
    assert x % y == 0
    return x // y


def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def remove_after_endoftext(text):
    """Removes everything after the first instance of "<|endoftext|>" in a string.

    Args:
      text: The string to process.

    Returns:
      The string with everything after the first "<|endoftext|>" removed.
    """
    endoftext_index = text.find("<|endoftext|>")
    if endoftext_index != -1:
        return text[: endoftext_index + len("<|endoftext|>")]
    else:
        return text


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


class TranscriptReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.ext = file_path.split(".")[-1]

    def read_vtt(self, file_path: str) -> Union[None, Tuple[Dict, str, str]]:
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
        if self.ext == "vtt":
            return self.read_vtt(self.file_path)
        elif self.ext == "srt":
            return self.read_srt(self.file_path)
        else:
            raise ValueError("Unsupported file type")

    def extract_text(transcript: Dict) -> Optional[str]:
        transcript_text = ""
        for _, text in transcript.items():
            transcript_text += text.strip() + " "
        return transcript_text.strip()


def write_segment(
    timestamps: List, transcript: Optional[Dict], output_dir: str, ext: str
) -> None:
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
    # truth, predicted
    if pair[0] == "" and pair[1] == "":
        return 0.0
    elif pair[0] == "" and pair[1] != "":
        return (0.01 * len(pair[1].split())) * 100.0
    else:
        return jiwer.wer(pair[0], pair[1]) * 100.0


def average_wer(pair_list: List[Tuple[str, str]]) -> float:
    # remember that tuple or list has to be of the form (truth, predicted)
    return np.round(sum(map(calculate_wer, pair_list)) / len(pair_list))


def clean_text(
    text_list: Union[List[Tuple[str, str]], List],
    normalizer: str,
    remove_diacritics: bool = True,
    split_letters: bool = True,
) -> List[Tuple[str, str]]:
    if normalizer == "basic":
        normalizer = BasicTextNormalizer(
            remove_diacritics=remove_diacritics, split_letters=split_letters
        )
    elif normalizer == "english":
        normalizer = EnglishTextNormalizer()
    else:
        raise ValueError("Unsupported normalizer")

    if len(text_list[0]) == 2: # is tuple
        normalize = lambda pair: (
            normalizer.clean(pair[0]) if normalizer == "basic" else normalizer(pair[0]),
            normalizer.clean(pair[1]) if normalizer == "basic" else normalizer(pair[1]),
        )
    else:
        normalize = lambda text: normalizer.clean(text) if normalizer == "basic" else normalizer(text)
    
    return list(map(normalize, text_list))
