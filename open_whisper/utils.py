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

    if len(text_list[0]) == 2:  # is tuple
        normalize = lambda pair: (
            normalizer.clean(pair[0]) if normalizer == "basic" else normalizer(pair[0]),
            normalizer.clean(pair[1]) if normalizer == "basic" else normalizer(pair[1]),
        )
    else:
        normalize = lambda text: (
            normalizer.clean(text) if normalizer == "basic" else normalizer(text)
        )

    return list(map(normalize, text_list))


import json
import os
import re
import sys
import zlib
from typing import Callable, List, Optional, TextIO

system_encoding = sys.getdefaultencoding()

if system_encoding != "utf-8":

    def make_safe(string):
        # replaces any character not representable using the system default encoding with an '?',
        # avoiding UnicodeEncodeError (https://github.com/openai/whisper/discussions/729).
        return string.encode(system_encoding, errors="replace").decode(system_encoding)

else:

    def make_safe(string):
        # utf-8 can encode any Unicode code point, so no need to do the round-trip encoding
        return string


def exact_div(x, y):
    assert x % y == 0
    return x // y


def str2bool(string):
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def optional_int(string):
    return None if string == "None" else int(string)


def optional_float(string):
    return None if string == "None" else float(string)


def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def get_start(segments: List[dict]) -> Optional[float]:
    return next(
        (w["start"] for s in segments for w in s["words"]),
        segments[0]["start"] if segments else None,
    )


def get_end(segments: List[dict]) -> Optional[float]:
    return next(
        (w["end"] for s in reversed(segments) for w in reversed(s["words"])),
        segments[-1]["end"] if segments else None,
    )


class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(
        self, result: dict, audio_path: str, options: Optional[dict] = None, **kwargs
    ):
        audio_basename = os.path.basename(audio_path)
        audio_basename = os.path.splitext(audio_basename)[0]
        output_path = os.path.join(
            self.output_dir, audio_basename + "." + self.extension
        )

        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f, options=options, **kwargs)

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        raise NotImplementedError


class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        for segment in result["segments"]:
            print(segment["text"].strip(), file=file, flush=True)


class SubtitlesWriter(ResultWriter):
    always_include_hours: bool
    decimal_marker: str

    def iterate_result(
        self,
        result: dict,
        options: Optional[dict] = None,
        *,
        max_line_width: Optional[int] = None,
        max_line_count: Optional[int] = None,
        highlight_words: bool = False,
        max_words_per_line: Optional[int] = None,
    ):
        options = options or {}
        max_line_width = max_line_width or options.get("max_line_width")
        max_line_count = max_line_count or options.get("max_line_count")
        highlight_words = highlight_words or options.get("highlight_words", False)
        max_words_per_line = max_words_per_line or options.get("max_words_per_line")
        preserve_segments = max_line_count is None or max_line_width is None
        max_line_width = max_line_width or 1000
        max_words_per_line = max_words_per_line or 1000

        def iterate_subtitles():
            line_len = 0
            line_count = 1
            # the next subtitle to yield (a list of word timings with whitespace)
            subtitle: List[dict] = []
            last: float = get_start(result["segments"]) or 0.0
            for segment in result["segments"]:
                chunk_index = 0
                words_count = max_words_per_line
                while chunk_index < len(segment["words"]):
                    remaining_words = len(segment["words"]) - chunk_index
                    if max_words_per_line > len(segment["words"]) - chunk_index:
                        words_count = remaining_words
                    for i, original_timing in enumerate(
                        segment["words"][chunk_index : chunk_index + words_count]
                    ):
                        timing = original_timing.copy()
                        long_pause = (
                            not preserve_segments and timing["start"] - last > 3.0
                        )
                        has_room = line_len + len(timing["word"]) <= max_line_width
                        seg_break = i == 0 and len(subtitle) > 0 and preserve_segments
                        if (
                            line_len > 0
                            and has_room
                            and not long_pause
                            and not seg_break
                        ):
                            # line continuation
                            line_len += len(timing["word"])
                        else:
                            # new line
                            timing["word"] = timing["word"].strip()
                            if (
                                len(subtitle) > 0
                                and max_line_count is not None
                                and (long_pause or line_count >= max_line_count)
                                or seg_break
                            ):
                                # subtitle break
                                yield subtitle
                                subtitle = []
                                line_count = 1
                            elif line_len > 0:
                                # line break
                                line_count += 1
                                timing["word"] = "\n" + timing["word"]
                            line_len = len(timing["word"].strip())
                        subtitle.append(timing)
                        last = timing["start"]
                    chunk_index += max_words_per_line
            if len(subtitle) > 0:
                yield subtitle

        if len(result["segments"]) > 0 and "words" in result["segments"][0]:
            for subtitle in iterate_subtitles():
                subtitle_start = self.format_timestamp(subtitle[0]["start"])
                subtitle_end = self.format_timestamp(subtitle[-1]["end"])
                subtitle_text = "".join([word["word"] for word in subtitle])
                if highlight_words:
                    last = subtitle_start
                    all_words = [timing["word"] for timing in subtitle]
                    for i, this_word in enumerate(subtitle):
                        start = self.format_timestamp(this_word["start"])
                        end = self.format_timestamp(this_word["end"])
                        if last != start:
                            yield last, start, subtitle_text

                        yield start, end, "".join(
                            [
                                (
                                    re.sub(r"^(\s*)(.*)$", r"\1<u>\2</u>", word)
                                    if j == i
                                    else word
                                )
                                for j, word in enumerate(all_words)
                            ]
                        )
                        last = end
                else:
                    yield subtitle_start, subtitle_end, subtitle_text
        else:
            for segment in result["segments"]:
                segment_start = self.format_timestamp(segment["start"])
                segment_end = self.format_timestamp(segment["end"])
                segment_text = segment["text"].strip().replace("-->", "->")
                yield segment_start, segment_end, segment_text

    def format_timestamp(self, seconds: float):
        return format_timestamp(
            seconds=seconds,
            always_include_hours=self.always_include_hours,
            decimal_marker=self.decimal_marker,
        )


class WriteVTT(SubtitlesWriter):
    extension: str = "vtt"
    always_include_hours: bool = False
    decimal_marker: str = "."

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        print("WEBVTT\n", file=file)
        for start, end, text in self.iterate_result(result, options, **kwargs):
            print(f"{start} --> {end}\n{text}\n", file=file, flush=True)


class WriteSRT(SubtitlesWriter):
    extension: str = "srt"
    always_include_hours: bool = True
    decimal_marker: str = ","

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        for i, (start, end, text) in enumerate(
            self.iterate_result(result, options, **kwargs), start=1
        ):
            print(f"{i}\n{start} --> {end}\n{text}\n", file=file, flush=True)


class WriteTSV(ResultWriter):
    """
    Write a transcript to a file in TSV (tab-separated values) format containing lines like:
    <start time in integer milliseconds>\t<end time in integer milliseconds>\t<transcript text>

    Using integer milliseconds as start and end times means there's no chance of interference from
    an environment setting a language encoding that causes the decimal in a floating point number
    to appear as a comma; also is faster and more efficient to parse & store, e.g., in C++.
    """

    extension: str = "tsv"

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        print("start", "end", "text", sep="\t", file=file)
        for segment in result["segments"]:
            print(round(1000 * segment["start"]), file=file, end="\t")
            print(round(1000 * segment["end"]), file=file, end="\t")
            print(segment["text"].strip().replace("\t", " "), file=file, flush=True)


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        json.dump(result, file)


def get_writer(
    output_format: str, output_dir: str
) -> Callable[[dict, TextIO, dict], None]:
    writers = {
        "txt": WriteTXT,
        "vtt": WriteVTT,
        "srt": WriteSRT,
        "tsv": WriteTSV,
        "json": WriteJSON,
    }

    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writers.values()]

        def write_all(
            result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
        ):
            for writer in all_writers:
                writer(result, file, options, **kwargs)

        return write_all

    return writers[output_format](output_dir)
