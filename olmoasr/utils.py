from datetime import datetime, timedelta
from typing import Dict, Union, Tuple, List, Optional, Any
import subprocess
from subprocess import CalledProcessError
import numpy as np
import tarfile
import gzip
import json
import pysrt
import webvtt
import jiwer
from whisper.tokenizer import get_tokenizer


def remove_after_endoftext(text: str) -> str:
    """Remove everything after the first instance of "<|endoftext|>" in a string.

    Args:
        text: The string to process

    Returns:
        The string with everything after the first "<|endoftext|>" removed.
        If no "<|endoftext|>" is found, returns the original string.
    """
    endoftext_index = text.find("<|endoftext|>")
    if endoftext_index != -1:
        return text[: endoftext_index + len("<|endoftext|>")]
    else:
        return text


def convert_to_milliseconds(timestamp: str) -> int:
    """Convert a timestamp in HH:MM:SS.mmm format to milliseconds.

    Args:
        timestamp: Timestamp in the format HH:MM:SS.mmm

    Returns:
        Timestamp in milliseconds

    Raises:
        ValueError: If timestamp format is invalid
    """
    try:
        h, m, s, ms = map(float, timestamp.replace(".", ":").split(":"))
        return int(h * 3600000 + m * 60000 + s * 1000 + ms)
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid timestamp format: {timestamp}") from e


def calculate_difference(timestamp1: str, timestamp2: str) -> int:
    """Calculate the difference between two timestamps in milliseconds.

    Args:
        timestamp1: First timestamp in HH:MM:SS.mmm format
        timestamp2: Second timestamp in HH:MM:SS.mmm format

    Returns:
        Difference between the two timestamps in milliseconds

    Raises:
        ValueError: If second timestamp is earlier than first timestamp
    """
    time1 = convert_to_milliseconds(timestamp1)
    time2 = convert_to_milliseconds(timestamp2)
    if time2 < time1:
        raise ValueError(
            "Second timestamp is less than the first timestamp. Needs to be greater than the first timestamp."
        )
    return time2 - time1


def adjust_timestamp(timestamp: str, milliseconds: int) -> str:
    """Adjust a timestamp by a specified number of milliseconds.

    Args:
        timestamp: Timestamp in HH:MM:SS.mmm format
        milliseconds: Number of milliseconds to add (positive) or subtract (negative)

    Returns:
        Adjusted timestamp in HH:MM:SS.mmm format

    Raises:
        ValueError: If timestamp format is invalid
    """
    try:
        # Convert the HH:MM:SS.mmm format to a datetime object
        original_time = datetime.strptime(timestamp, "%H:%M:%S.%f")

        # Adjust the time by the specified number of milliseconds
        adjusted_time = original_time + timedelta(milliseconds=milliseconds)

        # Convert back to the HH:MM:SS.mmm string format
        return adjusted_time.strftime("%H:%M:%S.%f")[
            :-3
        ]  # Truncate microseconds to milliseconds
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format: {timestamp}") from e


def trim_audio(
    audio_file: str,
    start: str,
    end: str,
    output_dir: str,
    in_memory: bool,
    sample_rate: int = 16000,
    start_window: int = 0,
    end_window: int = 0,
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Trim an audio file to specified start and end timestamps.

    Trims the audio file to the specified start and end timestamps and optionally saves
    the trimmed audio file to the output directory. Window parameters allow extending
    the trimming boundaries.

    Args:
        audio_file: Path to the audio file
        start: Start timestamp in HH:MM:SS.mmm format
        end: End timestamp in HH:MM:SS.mmm format
        output_dir: Directory to save the trimmed audio file
        in_memory: Whether to return audio data in memory only
        sample_rate: Audio sample rate in Hz (default: 16000)
        start_window: Seconds to extend before start timestamp (positive) or after (negative)
        end_window: Seconds to extend after end timestamp (positive) or before (negative)

    Returns:
        Tuple of (output_file_path, audio_array). Returns (None, None) if ffmpeg fails.
    """
    # Apply window adjustments to timestamps
    adjusted_start = start
    adjusted_end = end

    if start_window != 0:
        adjusted_start = adjust_timestamp(start, start_window * 1000)

    if end_window != 0:
        adjusted_end = adjust_timestamp(end, end_window * 1000)

    output_file = f"{output_dir}/{adjusted_start.replace('.', ',')}_{adjusted_end.replace('.', ',')}.npy"

    # Build ffmpeg command
    command = [
        "ffmpeg",
        "-threads",
        "0",
        "-i",
        audio_file,
        "-ss",
        adjusted_start,
        "-to",
        adjusted_end,
        "-c:a",
        "pcm_s16le",
        "-filter_complex",
        f"aresample={sample_rate},pan=mono|c0=c0",
        "-f",
        "s16le",
        "-",
    ]

    try:
        result = subprocess.run(command, capture_output=True, check=True)
        audio_arr = np.frombuffer(result.stdout, np.int16).flatten()

        if not in_memory:
            np.save(output_file, audio_arr)

        return output_file, audio_arr
    except CalledProcessError:
        return None, None


class TranscriptReader:
    """A class to read WebVTT or SRT transcript files and extract transcript data.

    Attributes:
        file_path: Path to the transcript file (optional)
        transcript_string: Transcript content as string (optional)
        ext: File extension ('vtt' or 'srt')
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        transcript_string: Optional[str] = None,
        ext: Optional[str] = None,
    ):
        """Initialize TranscriptReader with either file path or transcript string.

        Args:
            file_path: Path to transcript file
            transcript_string: Transcript content as string
            ext: File extension ('vtt' or 'srt') - required if using transcript_string

        Raises:
            ValueError: If neither file_path nor transcript_string is provided
        """
        if file_path is None and transcript_string is None:
            raise ValueError("Either file_path or transcript_string must be provided")

        self.file_path = file_path
        self.transcript_string = transcript_string

        if file_path is not None:
            self.ext = file_path.split(".")[-1]
        else:
            self.ext = ext

    def _format_timestamp(
        self, hours: int, minutes: int, seconds: int, milliseconds: int
    ) -> str:
        """Format timestamp components into HH:MM:SS.mmm format."""
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

    def _read_transcript_file(
        self, file_type: str
    ) -> Tuple[Dict[Tuple[str, str], str], str, str]:
        """Read transcript file and extract timing and text data.

        Args:
            file_type: Type of transcript file ('vtt' or 'srt')

        Returns:
            Tuple of (transcript_dict, start_timestamp, end_timestamp)
        """
        transcript = {}

        if file_type == "vtt":
            if self.file_path is not None:
                captions = webvtt.read(self.file_path)
            else:
                if self.transcript_string is None:
                    raise ValueError(
                        "transcript_string cannot be None when file_path is not provided"
                    )
                captions = webvtt.from_string(self.transcript_string)

            if len(captions) == 0:
                return transcript, "", ""

            transcript_start = captions[0].start
            transcript_end = captions[-1].end

            for caption in captions:
                transcript[(caption.start, caption.end)] = caption.text

        elif file_type == "srt":
            if self.file_path is not None:
                subs = pysrt.open(self.file_path)
            else:
                if self.transcript_string is None:
                    raise ValueError(
                        "transcript_string cannot be None when file_path is not provided"
                    )
                subs = pysrt.from_string(self.transcript_string)

            if len(subs) == 0:
                return transcript, "", ""

            transcript_start = self._format_timestamp(
                subs[0].start.hours,
                subs[0].start.minutes,
                subs[0].start.seconds,
                subs[0].start.milliseconds,
            )
            transcript_end = self._format_timestamp(
                subs[-1].end.hours,
                subs[-1].end.minutes,
                subs[-1].end.seconds,
                subs[-1].end.milliseconds,
            )

            for sub in subs:
                start = self._format_timestamp(
                    sub.start.hours,
                    sub.start.minutes,
                    sub.start.seconds,
                    sub.start.milliseconds,
                )
                end = self._format_timestamp(
                    sub.end.hours,
                    sub.end.minutes,
                    sub.end.seconds,
                    sub.end.milliseconds,
                )
                transcript[(start, end)] = sub.text

        return transcript, transcript_start, transcript_end

    def read_vtt(
        self, file_path: Optional[str], transcript_string: Optional[str]
    ) -> Tuple[Dict[Tuple[str, str], str], str, str]:
        """Read a WebVTT file.

        Args:
            file_path: Path to the WebVTT file
            transcript_string: WebVTT content as string

        Returns:
            Tuple of (transcript_dict, start_timestamp, end_timestamp)
        """
        return self._read_transcript_file("vtt")

    def read_srt(
        self, file_path: Optional[str], transcript_string: Optional[str]
    ) -> Tuple[Dict[Tuple[str, str], str], str, str]:
        """Read an SRT file or string.

        Args:
            file_path: Path to the SRT file
            transcript_string: SRT transcript as a string

        Returns:
            Tuple of (transcript_dict, start_timestamp, end_timestamp)
        """
        return self._read_transcript_file("srt")

    def read(self) -> Tuple[Dict[Tuple[str, str], str], str, str]:
        """Read the transcript file based on its extension.

        Returns:
            Tuple of (transcript_dict, start_timestamp, end_timestamp)

        Raises:
            ValueError: If file extension is not supported
        """
        if self.ext == "vtt":
            return self.read_vtt(self.file_path, self.transcript_string)
        elif self.ext == "srt":
            return self.read_srt(self.file_path, self.transcript_string)
        else:
            raise ValueError(f"Unsupported file type: {self.ext}")

    def extract_text(self, transcript: Dict[Tuple[str, str], str]) -> str:
        """Extract the text from the transcript dictionary.

        Args:
            transcript: Transcript as a dictionary mapping (start, end) tuples to text

        Returns:
            The extracted text with segments joined by spaces
        """
        if not transcript:
            return ""

        transcript_text = ""
        for _, text in transcript.items():
            transcript_text += text.strip() + " "
        return transcript_text.strip()


def write_segment(
    audio_begin: str,
    timestamps: List[Tuple[str, str]],
    transcript: Optional[Dict[Tuple[str, str], str]],
    output_dir: str,
    ext: str,
    in_memory: bool,
) -> Tuple[str, str, str, bool]:
    """Write a segment of the transcript to a file.

    Args:
        audio_begin: Beginning timestamp of the audio segment
        timestamps: List of (start, end) timestamp tuples
        transcript: Transcript dictionary mapping timestamps to text
        output_dir: Directory to save the transcript file
        ext: File extension ('vtt' or 'srt')
        in_memory: Whether to save the transcript in memory only

    Returns:
        Tuple of (output_file_path, transcript_string, end_timestamp, only_no_ts_mode)
    """
    only_no_ts_mode = False
    audio_begin_ms = convert_to_milliseconds(audio_begin)

    output_file = f"{output_dir}/{audio_begin.replace('.', ',')}_{timestamps[-1][1].replace('.', ',')}.{ext}"
    transcript_string = ""

    if ext == "vtt":
        transcript_string += "WEBVTT\n\n"

    if transcript is None:
        if not in_memory:
            with open(output_file, "w") as f:
                f.write(transcript_string)
        return output_file, transcript_string, "", only_no_ts_mode

    for i, (start_ts, end_ts) in enumerate(timestamps):
        start_ms = convert_to_milliseconds(start_ts)
        end_ms = convert_to_milliseconds(end_ts)

        # Check if timestamps are before audio begin
        if start_ms < audio_begin_ms or end_ms < audio_begin_ms:
            only_no_ts_mode = True

        # Adjust timestamps relative to audio begin
        start = adjust_timestamp(start_ts, -audio_begin_ms)
        end = adjust_timestamp(end_ts, -audio_begin_ms)

        # Format timestamps based on file type
        if ext == "srt":
            start = start.replace(".", ",")
            end = end.replace(".", ",")
            transcript_string += f"{i + 1}\n"

        # Add transcript text
        text = transcript.get((start_ts, end_ts), "")
        transcript_string += f"{start} --> {end}\n{text}\n\n"

    if not in_memory:
        with open(output_file, "w") as f:
            f.write(transcript_string)

    return output_file, transcript_string, end.replace(",", "."), only_no_ts_mode


def calculate_wer(pair: Tuple[str, str]) -> float:
    """Calculate the Word Error Rate (WER) between two strings.

    Args:
        pair: Tuple of (reference_text, hypothesis_text)

    Returns:
        Word Error Rate as a percentage (0.0 to 100.0)
    """
    reference, hypothesis = pair
    if reference.strip() == "":
        return 0.0
    return jiwer.wer(reference, hypothesis) * 100.0


def over_ctx_len(
    timestamps: List[Tuple[str, str]],
    transcript: Optional[Dict[Tuple[str, str], str]],
    language: Optional[str] = None,
    last_seg: bool = False,
) -> Tuple[bool, Optional[Union[str, Dict[str, Union[bool, int]]]]]:
    """Check if transcript text exceeds model context length.

    Args:
        timestamps: List of (start, end) timestamp tuples
        transcript: Transcript dictionary mapping timestamps to text
        language: Language code for tokenizer selection
        last_seg: Whether this is the last segment

    Returns:
        Tuple of (exceeds_limit, mode_info)
        - exceeds_limit: True if text exceeds context length in both modes
        - mode_info: None if exceeds_limit is True, otherwise dict with mode availability
    """
    try:
        if transcript is None:
            return True, None

        # Get appropriate tokenizer
        if language is None:
            tokenizer = get_tokenizer(multilingual=False)
        else:
            tokenizer = get_tokenizer(language=language, multilingual=True)

        # Tokenize text segments
        text_tokens = []
        for start_ts, end_ts in timestamps:
            text = transcript.get((start_ts, end_ts), "")
            tokens = tokenizer.encode(" " + text.strip())
            text_tokens.append(tokens)

        # Calculate token counts
        num_timestamp_tokens = (len(timestamps) * 2) + (0 if last_seg else 1)
        num_text_tokens = sum(len(token_group) for token_group in text_tokens)
        num_tokens_ts_mode = num_timestamp_tokens + num_text_tokens + 2  # sot + eot
        num_tokens_no_ts_mode = num_text_tokens + 3  # sot + notimestamps + eot

        # Check context length limits
        context_limit = 448

        if num_tokens_ts_mode > context_limit and num_tokens_no_ts_mode > context_limit:
            return True, None

        # Return mode availability information
        mode_info = {
            "ts_mode": num_tokens_ts_mode <= context_limit,
            "no_ts_mode": num_tokens_no_ts_mode <= context_limit,
            "num_tokens_no_ts_mode": num_tokens_no_ts_mode,
            "num_tokens_ts_mode": num_tokens_ts_mode,
        }

        return False, mode_info

    except (RuntimeError, Exception):
        return True, "error"


def timestamps_valid(
    timestamps: List[Tuple[str, str]], global_start: str, global_end: str
) -> bool:
    """Validate that timestamps are within bounds and properly ordered.

    Args:
        timestamps: List of (start, end) timestamp tuples
        global_start: Global start timestamp boundary
        global_end: Global end timestamp boundary

    Returns:
        True if all timestamps are valid, False otherwise
    """
    if not timestamps:
        return False

    # Convert to milliseconds for comparison
    to_ms = convert_to_milliseconds
    start_ms = to_ms(timestamps[0][0])
    end_ms = to_ms(timestamps[-1][1])
    g_start_ms = to_ms(global_start)
    g_end_ms = to_ms(global_end)

    # Check global bounds
    if start_ms < g_start_ms or end_ms > g_end_ms:
        return False

    # Check individual timestamp validity
    for start_ts, end_ts in timestamps:
        start_ms_seg = to_ms(start_ts)
        end_ms_seg = to_ms(end_ts)

        # Check segment validity and bounds
        if (
            start_ms_seg > end_ms_seg
            or start_ms_seg < g_start_ms
            or end_ms_seg > g_end_ms
            or start_ms_seg < start_ms
            or end_ms_seg > end_ms
        ):
            return False

    return True


def too_short_audio(audio_arr: np.ndarray, sample_rate: int = 16000) -> bool:
    """Check if audio array duration is too short.

    Args:
        audio_arr: Audio data as numpy array
        sample_rate: Audio sample rate in Hz

    Returns:
        True if audio duration is less than 15ms, False otherwise
    """
    duration = len(audio_arr) / sample_rate
    return duration < 0.015


def too_short_audio_text(start: str, end: str) -> bool:
    """Check if audio duration between timestamps is too short.

    Args:
        start: Start timestamp in HH:MM:SS.mmm format
        end: End timestamp in HH:MM:SS.mmm format

    Returns:
        True if duration is less than 15ms, False otherwise
    """
    duration = calculate_difference(start, end) / 1000  # Convert to seconds
    return duration < 0.015


class Segment:
    """
    Represents a transcript segment with associated metadata.

    This class encapsulates all information about a single segment including
    content, timing, file paths, and processing modes.
    """

    def __init__(
        self,
        subtitle_file: str,
        seg_content: str,
        text_timestamp: str,
        audio_timestamp: str,
        norm_end: float,
        video_id: str,
        seg_id: str,
        audio_file: str,
        ts_mode: bool,
        no_ts_mode: bool,
        only_no_ts_mode: bool,
        num_tokens_no_ts_mode: int,
        num_tokens_ts_mode: int,
    ) -> None:
        """
        Initialize a Segment object.

        Args:
            subtitle_file: Path to the subtitle file
            seg_content: Content of the segment
            text_timestamp: Timestamp of the segment in text format
            audio_timestamp: Timestamp of the audio segment
            norm_end: Normalized end time of the segment in milliseconds
            video_id: Unique identifier for the video
            seg_id: Unique identifier for the segment
            audio_file: Path to the audio file
            ts_mode: Whether the segment supports timestamp mode
            no_ts_mode: Whether the segment supports no timestamp mode
            only_no_ts_mode: Whether the segment only supports no timestamp mode
            num_tokens_no_ts_mode: Number of tokens in no timestamp mode
            num_tokens_ts_mode: Number of tokens in timestamp mode
        """
        self.subtitle_file = subtitle_file
        self.audio_file = audio_file
        self.seg_content = seg_content
        self.text_timestamp = text_timestamp
        self.audio_timestamp = audio_timestamp
        self.norm_end = norm_end
        self.video_id = video_id
        self.seg_id = seg_id
        self.ts_mode = ts_mode
        self.no_ts_mode = no_ts_mode
        self.only_no_ts_mode = only_no_ts_mode
        self.num_tokens_no_ts_mode = num_tokens_no_ts_mode
        self.num_tokens_ts_mode = num_tokens_ts_mode

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the segment to a dictionary representation.

        Returns:
            Dictionary containing all segment data
        """
        return {
            "subtitle_file": self.subtitle_file,
            "seg_content": self.seg_content,
            "text_timestamp": self.text_timestamp,
            "audio_timestamp": self.audio_timestamp,
            "norm_end": self.norm_end,
            "id": self.video_id,
            "seg_id": self.seg_id,
            "audio_file": self.audio_file,
            "ts_mode": self.ts_mode,
            "no_ts_mode": self.no_ts_mode,
            "only_no_ts_mode": self.only_no_ts_mode,
            "num_tokens_no_ts_mode": self.num_tokens_no_ts_mode,
            "num_tokens_ts_mode": self.num_tokens_ts_mode,
        }

    def add_attr(self, key: str, value: Any) -> None:
        """
        Add a new attribute to the segment.

        Args:
            key: The attribute name
            value: The attribute value
        """
        self.__setattr__(key, value)


class MachineSegment:
    """
    Represents a machine-generated transcript segment.

    This class is used for segments created by automated speech recognition systems
    and contains essential metadata for alignment with manual transcripts.
    """

    def __init__(
        self,
        subtitle_file: str,
        seg_content: str,
        timestamp: str,
        video_id: str,
        audio_file: str,
    ) -> None:
        """
        Initialize a MachineSegment object.

        Args:
            subtitle_file: Path to the subtitle file
            seg_content: Content of the segment
            timestamp: Timestamp of the segment
            video_id: Unique identifier for the video
            audio_file: Path to the audio file
        """
        self.subtitle_file = subtitle_file
        self.seg_content = seg_content
        self.timestamp = timestamp
        self.video_id = video_id
        self.audio_file = audio_file

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the machine segment to a dictionary representation.

        Returns:
            Dictionary containing all machine segment data
        """
        return {
            "subtitle_file": self.subtitle_file,
            "seg_content": self.seg_content,
            "timestamp": self.timestamp,
            "video_id": self.video_id,
            "audio_file": self.audio_file,
        }


class SegmentCounter:
    """
    Tracks statistics during segment processing.

    This class maintains counters for various types of segments and errors
    encountered during the preprocessing pipeline.
    """

    def __init__(
        self,
        segment_count: int = 0,
        over_30_line_segment_count: int = 0,
        bad_text_segment_count: int = 0,
        over_ctx_len_segment_count: int = 0,
        faulty_audio_segment_count: int = 0,
        failed_transcript_count: int = 0,
    ) -> None:
        """
        Initialize a SegmentCounter object.

        Args:
            segment_count: Total number of valid segments processed
            over_30_line_segment_count: Number of segments over 30 seconds
            bad_text_segment_count: Number of segments with bad text
            over_ctx_len_segment_count: Number of segments over context length
            faulty_audio_segment_count: Number of segments with faulty audio
            failed_transcript_count: Number of failed transcript processing attempts
        """
        self.segment_count = segment_count
        self.over_30_line_segment_count = over_30_line_segment_count
        self.bad_text_segment_count = bad_text_segment_count
        self.over_ctx_len_segment_count = over_ctx_len_segment_count
        self.faulty_audio_segment_count = faulty_audio_segment_count
        self.failed_transcript_count = failed_transcript_count

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the counter to a dictionary representation.

        Returns:
            Dictionary containing all counter values
        """
        return {
            "segment_count": self.segment_count,
            "over_30_line_segment_count": self.over_30_line_segment_count,
            "bad_text_segment_count": self.bad_text_segment_count,
            "over_ctx_len_segment_count": self.over_ctx_len_segment_count,
            "faulty_audio_segment_count": self.faulty_audio_segment_count,
            "failed_transcript_count": self.failed_transcript_count,
        }


def sum_counters(counters: List[SegmentCounter]) -> SegmentCounter:
    """
    Sum multiple SegmentCounter objects efficiently.

    Args:
        counters: List of SegmentCounter objects to sum

    Returns:
        New SegmentCounter object with summed values
    """
    if not counters:
        return SegmentCounter()

    return SegmentCounter(
        segment_count=sum(c.segment_count for c in counters if c is not None),
        over_30_line_segment_count=sum(
            c.over_30_line_segment_count for c in counters if c is not None
        ),
        bad_text_segment_count=sum(
            c.bad_text_segment_count for c in counters if c is not None
        ),
        over_ctx_len_segment_count=sum(
            c.over_ctx_len_segment_count for c in counters if c is not None
        ),
        faulty_audio_segment_count=sum(
            c.faulty_audio_segment_count for c in counters if c is not None
        ),
        failed_transcript_count=sum(
            c.failed_transcript_count for c in counters if c is not None
        ),
    )


def get_seg_text(segment: Segment) -> str:
    """
    Extract text content from a segment.

    Args:
        segment: Segment object containing transcript content

    Returns:
        Extracted text content from the segment, or empty string if extraction fails
    """
    try:
        reader = TranscriptReader(
            file_path=None,
            transcript_string=segment.seg_content,
            ext="vtt" if segment.seg_content.startswith("WEBVTT") else "srt",
        )
        result = reader.read()
        if result is None:
            return ""
        t_dict, *_ = result
        segment_text = reader.extract_text(t_dict)
        return segment_text.strip() if segment_text else ""
    except Exception:
        return ""


def get_mach_seg_text(mach_segment: MachineSegment) -> str:
    """
    Extract text content from a machine-generated segment.

    Args:
        mach_segment: MachineSegment object containing machine transcript content

    Returns:
        Extracted text content from the machine segment, or empty string if extraction fails
    """
    try:
        content = webvtt.from_string(mach_segment.seg_content)
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
        return mach_segment_text.strip()
    except Exception:
        return ""


def read_file_in_tar(
    tar_gz_path: str, file_in_tar: str, audio_dir: Optional[str]
) -> Optional[str]:
    """
    Read and extract a specific file from a .tar.gz archive.

    Args:
        tar_gz_path: Path to the .tar.gz archive file
        file_in_tar: Path to the file within the archive to extract
        audio_dir: Directory to save audio files (required for audio files)

    Returns:
        File content as string for text files, file path for audio files,
        or None for unsupported files or errors
    """
    try:
        # Open the tar.gz archive and extract the specified file
        with tarfile.open(tar_gz_path, "r:gz") as tar:
            extracted_file = tar.extractfile(file_in_tar)
            if extracted_file is None:
                return None
            binary_content = extracted_file.read()

            # Handle text-based transcript files (SRT/VTT)
            if file_in_tar.endswith("srt") or file_in_tar.endswith("vtt"):
                file_content = binary_content.decode("utf-8")
                return file_content

            # Handle audio files - save to disk and return path
            elif file_in_tar.endswith("m4a"):
                if audio_dir is None:
                    return None
                output_path = f"{audio_dir}/{file_in_tar.split('/')[-1]}"
                with open(output_path, "wb") as f:
                    f.write(binary_content)
                return output_path

            # Unsupported file type
            else:
                return None
    except Exception:
        return None


def unarchive_jsonl_gz(
    file_path: str, output_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Unarchive a .jsonl.gz file and optionally save the uncompressed content.

    Args:
        file_path: Path to the .jsonl.gz file
        output_path: Optional path to save the uncompressed .jsonl file

    Returns:
        List of JSON objects parsed from the .jsonl file

    Raises:
        FileNotFoundError: If the input file doesn't exist
        JSONDecodeError: If the file contains invalid JSON
    """
    data = []
    with gzip.open(file_path, "rt", encoding="utf-8") as gz_file:
        data = [json.loads(line.strip()) for line in gz_file]

    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_file:
            for json_obj in data:
                out_file.write(json.dumps(json_obj) + "\n")

    return data


# Legacy commented code preserved for reference
# def write_segment(
#     timestamps: List,
#     transcript: Optional[Dict],
#     output_dir: str,
#     ext: str,
#     in_memory: bool,
# ) -> Tuple[str, str]:
#     """Write a segment of the transcript to a file

#     Args:
#         timestamps: List of timestamps
#         transcript: Transcript as a dictionary
#         output_dir: Directory to save the transcript file
#         ext: File extension
#         in_memory: Whether to save the transcript in memory or to a file

#     Returns:
#         Path to the output transcript file
#     """
#     output_file = f"{output_dir}/{timestamps[0][0].replace('.', ',')}_{timestamps[-1][1].replace('.', ',')}.{ext}"
#     transcript_string = ""

#     if ext == "vtt":
#         transcript_string += "WEBVTT\n\n"

#     if transcript is None:
#         if not in_memory:
#             with open(output_file, "w") as f:
#                 f.write(transcript_string)
#         return output_file, transcript_string, ""

#     for i in range(len(timestamps)):
#         start = adjust_timestamp(
#             timestamp=timestamps[i][0],
#             milliseconds=-convert_to_milliseconds(timestamps[0][0]),
#         )
#         end = adjust_timestamp(
#             timestamp=timestamps[i][1],
#             milliseconds=-convert_to_milliseconds(timestamps[0][0]),
#         )

#         if ext == "srt":
#             start = start.replace(".", ",")
#             end = end.replace(".", ",")
#             transcript_string += f"{i + 1}\n"

#         transcript_string += (
#             f"{start} --> {end}\n{transcript[(timestamps[i][0], timestamps[i][1])]}\n\n"
#         )

#     if not in_memory:
#         with open(output_file, "w") as f:
#             f.write(transcript_string)

#     return output_file, transcript_string, end.replace(",", ".")
