from open_whisper.preprocess import (
    download_audio,
    download_transcript,
)
import multiprocessing
from tqdm import tqdm
from itertools import repeat
from typing import List
import shutil
import tarfile
import subprocess
from moto3.queue_manager import QueueManager
from moto3.s3_manager import S3Manager
import time

qm = QueueManager("whisper-downloading")
s3m = S3Manager("mattd-public/whisper")

def download(
    video_id: str, output_dir: str, audio_ext: str, lang_code: str, sub_format: str
) -> None:
    """Download audio and transcript files for a given video

    Args:
        video_id: Video ID to download
        output_dir: Directory to store the downloaded files
        audio_ext: Audio file extension
        lang_code: Language code of the transcript
        sub_format: Format of the transcript
    """
    download_audio(video_id=video_id, output_dir=output_dir, ext=audio_ext)
    download_transcript(
        video_id=video_id,
        lang_code=lang_code,
        output_dir=output_dir,
        sub_format=sub_format,
    )


def parallel_download(args) -> None:
    """Parallel download of audio and transcript files"""
    download(*args)


def compress_dir(source_dir, tar_file):
    with tarfile.open(tar_file, "w:gz") as tar:
        tar.add(source_dir, arcname=".")


def download_to_s3(id_lang: List[List], group_id: str) -> None:
    """Download transcript and audio files

    Download transcript and audio files from a list of video IDs and language codes.

    Args:
        id_lang: List of video IDs and language codes
        group_id: Folder and name of tar file to save to
    """
    transcript_ext = "srt"
    audio_ext = "m4a"

    sample_id, sample_lang = zip(*id_lang)

    with multiprocessing.Pool(multiprocessing.cpu_count() * 7) as pool:
        out = list(
            tqdm(
                pool.imap_unordered(
                    parallel_download,
                    zip(
                        sample_id,
                        repeat(group_id),
                        repeat(audio_ext),
                        sample_lang,
                        repeat(transcript_ext),
                    ),
                ),
                total=len(sample_id),
            )
        )

    # compress files
    compress_dir(group_id, f"{group_id}.tar.gz")

    shutil.rmtree(group_id)


def main():
    while True:
        try:
            message, item = qm.get_next(visibility_timeout=(60*60)) # 1 hour visibility timeout

            id_lang = item["id_lang"]
            group_id = item["group_id"]

            download_to_s3(id_lang=id_lang, group_id=group_id)
            
            # upload metadata data about unavailable videos
            s3m.upload_file(f"metadata/{group_id}/unavailable_videos.txt", f"metadata/{group_id}/unavailable_videos.txt")

            qm.delete(message)

            command = [
                "python",
                "scripts/data/data_transfer/upload_to_s3.py",
                f"--group_id={group_id}",
            ]

            proc = subprocess.Popen(command)
            stdout, stderr = proc.communicate()
        except IndexError: # no more items in queue to process
            break
    

if __name__ == "__main__":
    main()
