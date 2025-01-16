import multiprocessing
from tqdm import tqdm
import glob
import os
import json
from fire import Fire
from typing import Optional, Tuple, Union, Dict
import webvtt
import pysrt
import gzip

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = "AKIASHLPW4FE63DTIAPD"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
import subprocess


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


class TranscriptReader:
    """A class to read in a WebVTT or SRT transcript file and extract the transcript

    Attributes:
        file_path: Path to the transcript file
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        transcript_string: Optional[str] = None,
        ext: Optional[str] = None,
    ):
        if file_path is None and transcript_string is None:
            raise ValueError("Either file_path or transcript_string must be provided")

        if file_path is not None:
            self.file_path = file_path
            self.ext = file_path.split(".")[-1]
            self.transcript_string = transcript_string
        elif transcript_string is not None:
            self.transcript_string = transcript_string
            self.ext = ext
            self.file_path = file_path

    def read_vtt(
        self, file_path: Optional[str], transcript_string: Optional[str]
    ) -> Union[None, Tuple[Dict, str, str]]:
        """Read a WebVTT file

        Args:
            file_path: Path to the WebVTT file

        Returns:
            A tuple containing the transcript, start timestamp, and end timestamp or None if the file is empty
        """
        transcript = {}
        if file_path is not None:
            captions = webvtt.read(file_path)
        elif transcript_string is not None:
            captions = webvtt.from_string(transcript_string)

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

    def read_srt(
        self, file_path: Optional[str], transcript_string: Optional[str]
    ) -> Union[None, Tuple[Dict, str, str]]:
        """Read an SRT file or string

        Args:
            file_path: Path to the SRT file
            transcript_string: SRT transcript as a string

        Returns:
            A tuple containing the transcript, start timestamp, and end timestamp or None if the file is empty
        """
        transcript = {}
        if file_path is not None:
            subs = pysrt.open(file_path)
        elif transcript_string is not None:
            subs = pysrt.from_string(transcript_string)

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
            return self.read_vtt(
                file_path=self.file_path, transcript_string=self.transcript_string
            )
        elif self.ext == "srt":
            return self.read_srt(
                file_path=self.file_path, transcript_string=self.transcript_string
            )
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


def to_dicts(file_path: str):
    content = open(file_path, "r").read()
    ext = file_path.split(".")[-1]
    reader = TranscriptReader(file_path=None, transcript_string=content, ext=ext)
    transcript, transcript_start, transcript_end = reader.read()
    if len(transcript) == 0:
        length = 0
    else:
        try:
            length = calculate_difference(transcript_start, transcript_end) / 1000
        except ValueError:
            length = 0

    return {
        "subtitle_file": file_path,
        "content": content,
        "length": length,
        "audio_file": file_path.split(".")[0] + ".m4a",
    }


def upload_to_s3(file_path: str, bucket: str, bucket_prefix: str):
    cmd = [
        "aws",
        "s3",
        "cp",
        file_path,
        f"s3://{bucket}/{bucket_prefix}/{os.path.basename(file_path)}",
    ]
    res = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(res.stderr)
        raise ValueError("Failed to upload to S3")

def main(
    data_dir: str,
    output_dir: str,
    batch_size: int,
    start_shard_idx: int,
    end_shard_idx: int,
    bucket: str,
    bucket_prefix: str,
):
    os.makedirs(output_dir, exist_ok=True)
    shard_dirs = sorted(glob.glob(data_dir + "/*"))[start_shard_idx : end_shard_idx + 1]
    batch_idx = int(os.getenv("BEAKER_REPLICA_RANK"))
    shard_dirs = shard_dirs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    print(f"{len(shard_dirs)=}")
    print(f"{shard_dirs[:5]=}")

    for shard_dir in tqdm(shard_dirs, total=len(shard_dirs)):
        if int(shard_dir.split("/")[-1]) < 2449:
            ext = "srt"
        else:
            ext = "vtt"

        transcript_files = glob.glob(shard_dir + f"/*/*.{ext}")
        print(f"{len(transcript_files)=}")
        print(f"{transcript_files[:5]=}")

        shard_idx = shard_dir.split('/')[-1]
        with multiprocessing.Pool() as pool:
            text_dicts = list(
                tqdm(
                    pool.imap_unordered(to_dicts, transcript_files),
                    total=len(transcript_files),
                )
            )

        print(f"{len(text_dicts)=}")
        print(f"{text_dicts[0]=}")
        
        print(f"Writing to {output_dir}/shard_{shard_idx}.jsonl.gz")

        # Writing to a compressed JSONL file
        with gzip.open(f"{output_dir}/shard_{shard_idx}.jsonl.gz", "wt", encoding="utf-8") as f:
            for d in text_dicts:
                f.write(json.dumps(d) + "\n")

        print(f"Uploading to S3")
        upload_to_s3(f"{output_dir}/shard_{shard_idx}.jsonl.gz", bucket, bucket_prefix)
        os.remove(f"{output_dir}/shard_{shard_idx}.jsonl.gz")


if __name__ == "__main__":
    Fire(main)
