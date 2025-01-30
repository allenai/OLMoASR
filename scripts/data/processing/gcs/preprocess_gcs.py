import os
import tarfile
import shutil
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
import time
import multiprocessing
from fire import Fire
from itertools import repeat, chain
import numpy as np
from io import BytesIO
from google.cloud import storage
import sys
import glob
import json
import gzip

sys.path.append(os.getcwd())
import utils
import logging

os.environ["AWS_DEFAULT_REGION"] = "us-west-1"
os.environ["AWS_ACCESS_KEY_ID"] = "AKIASHLPW4FE63DTIAPD"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
from moto3.queue_manager import QueueManager
from google.cloud import compute_v1
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.FileHandler("preprocess_gcs.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


TRANSCRIPT_EXT = ".vtt"  # only applies for files after 1st round of downloading
AUDIO_EXT = ".m4a"
SEGMENT_COUNT_THRESHOLD = 120


def write_to_tar(
    shard: str, segments: Tuple[str, str, str, np.ndarray], seg_dir: str
) -> None:
    tar_name = f"{shard}.tar.gz"
    os.makedirs(seg_dir, exist_ok=True)
    tar_path = os.path.join(seg_dir, tar_name)

    with tarfile.open(tar_path, "w:gz") as tar:
        for segment in tqdm(segments, total=len(segments)):
            t_output_file, transcript_string, a_output_file, audio_arr = segment
            # Adding transcript to tar
            transcript_buffer = BytesIO()
            transcript_buffer.write(transcript_string.encode("utf-8"))
            transcript_buffer.seek(0)
            tarinfo_transcript = tarfile.TarInfo(
                name="/".join(t_output_file.split("/")[-2:])
            )
            tarinfo_transcript.size = transcript_buffer.getbuffer().nbytes
            tar.addfile(tarinfo_transcript, transcript_buffer)

            # Adding audio array to tar
            audio_buffer = BytesIO()
            np.save(audio_buffer, audio_arr)
            audio_buffer.seek(0)
            tarinfo_audio = tarfile.TarInfo(
                name="/".join(a_output_file.split("/")[-2:])
            )
            tarinfo_audio.size = audio_buffer.getbuffer().nbytes
            tar.addfile(tarinfo_audio, audio_buffer)

    return tar_path


def write_to_tar_text(shard: str, segments: Tuple[str, str], seg_dir: str) -> None:
    tar_name = f"{shard}.tar.gz"
    os.makedirs(seg_dir, exist_ok=True)
    tar_path = os.path.join(seg_dir, tar_name)

    with tarfile.open(tar_path, "w:gz") as tar:
        for segment in tqdm(segments, total=len(segments)):
            t_output_file, transcript_string = segment
            logger.info(f"{t_output_file=}")
            # Adding transcript to tar
            transcript_buffer = BytesIO()
            transcript_buffer.write(transcript_string.encode("utf-8"))
            transcript_buffer.seek(0)
            tarinfo_transcript = tarfile.TarInfo(
                name="/".join(t_output_file.split("/")[-2:])
            )
            tarinfo_transcript.size = transcript_buffer.getbuffer().nbytes
            tar.addfile(tarinfo_transcript, transcript_buffer)

    return tar_path


def download_from_gcs(bucket_name, gcs_file_name, local_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_file_name)
    blob.download_to_filename(local_file_name)


def list_files_with_extension(tar_gz_path, ext):
    """Lists all file names with a specific extension in a .tar.gz file."""
    # Open the tar.gz file
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        # Iterate over each file in the archive
        matching_files = sorted(
            [member.name for member in tar.getmembers() if member.name.endswith(ext)],
        )

    return matching_files


def read_file_in_tar(
    tar_gz_path, file_in_tar: str, audio_dir: Optional[str]
) -> Optional[str]:
    """Reads and returns the content of a specific file within a .tar.gz archive."""
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        binary_content = tar.extractfile(file_in_tar).read()
        if file_in_tar.endswith(TRANSCRIPT_EXT):
            file_content = binary_content.decode("utf-8")
            return file_content
        elif file_in_tar.endswith(AUDIO_EXT):
            output_path = f"{audio_dir}/{file_in_tar.split('/')[-1]}"
            with open(output_path, "wb") as f:
                f.write(binary_content)
            return output_path
        else:
            return None


def parallel_read_file_in_tar(args):
    return read_file_in_tar(*args)


def unarchive_jsonl_gz(file_path, output_path=None):
    """
    Unarchive a .jsonl.gz file and optionally save the uncompressed content to an output file.

    Args:
        file_path (str): Path to the .jsonl.gz file.
        output_path (str, optional): Path to save the uncompressed .jsonl file. Defaults to None.

    Returns:
        list[dict]: A list of JSON objects parsed from the .jsonl file.
    """
    data = []
    with gzip.open(file_path, "rt", encoding="utf-8") as gz_file:
        data = [json.loads(line.strip()) for line in gz_file]

    if output_path:
        with open(output_path, "w", encoding="utf-8") as out_file:
            for json_obj in data:
                out_file.write(json.dumps(json_obj) + "\n")

    return data


def chunk_audio_transcript(
    tar_gz_file: str,
    transcript_file: str,
    audio_file: str,
    log_dir: str,
    in_memory: bool = True,
) -> Optional[List[Tuple[str, str, str, np.ndarray]]]:
    os.makedirs(log_dir, exist_ok=True)

    try:
        video_id = transcript_file.split("/")[1]

        if "_" in tar_gz_file:
            language = tar_gz_file.split("_")[-1].split(".")[0]
        else:
            language = None

        transcript_string = read_file_in_tar(tar_gz_file, transcript_file, None)

        transcript_ext = transcript_file.split(".")[-1]
        segment_count = 0

        transcript, *_ = utils.TranscriptReader(
            file_path=None, transcript_string=transcript_string, ext=transcript_ext
        ).read()

        if len(transcript.keys()) == 0:
            with open(f"{log_dir}/empty_transcripts.txt", "a") as f:
                f.write(f"{video_id}\n")
            return None

        a = 0
        b = 0

        timestamps = list(transcript.keys())
        diff = 0
        init_diff = 0
        segments_list = []

        while a < len(transcript) + 1 and segment_count < SEGMENT_COUNT_THRESHOLD:
            init_diff = utils.calculate_difference(timestamps[a][0], timestamps[b][1])

            if init_diff < 30000:
                diff = init_diff
                b += 1
            else:
                # edge case (when transcript line is > 30s)
                if b == a:
                    with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                        f.write(f"{video_id}\tindex: {b}\n")

                    a += 1
                    b += 1

                    if a == b == len(transcript):
                        if segment_count == 0:
                            with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                                f.write(f"{video_id}\tdelete\n")
                        break

                    continue

                over_ctx_len, err = utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript, language=language
                )
                if not over_ctx_len:
                    t_output_file, transcript_string = utils.write_segment(
                        timestamps=timestamps[a:b],
                        transcript=transcript,
                        output_dir=video_id,
                        ext=transcript_ext,
                        in_memory=in_memory,
                    )

                    a_output_file, audio_arr = utils.trim_audio(
                        audio_file=audio_file,
                        start=timestamps[a][0],
                        end=timestamps[b - 1][1],
                        output_dir=video_id,
                        in_memory=in_memory,
                    )

                    if audio_arr is not None and not utils.too_short_audio(
                        audio_arr=audio_arr
                    ):
                        segments_list.append(
                            (t_output_file, transcript_string, a_output_file, audio_arr)
                        )
                        segment_count += 1
                    else:
                        if audio_arr is None:
                            with open(f"{log_dir}/faulty_audio.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                else:
                    if err is not None:
                        with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")
                    else:
                        with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")

                init_diff = 0
                diff = 0

                # checking for silence
                if timestamps[b][0] > timestamps[b - 1][1]:
                    silence_segments = (
                        utils.calculate_difference(
                            timestamps[b - 1][1], timestamps[b][0]
                        )
                        // 30000
                    )

                    for i in range(0, silence_segments + 1):
                        start = utils.adjust_timestamp(
                            timestamps[b - 1][1], (i * 30000)
                        )

                        if i == silence_segments:
                            if start == timestamps[b][0]:
                                continue
                            else:
                                end = timestamps[b][0]
                        else:
                            end = utils.adjust_timestamp(start, 30000)

                        t_output_file, transcript_string = utils.write_segment(
                            timestamps=[(start, end)],
                            transcript=None,
                            output_dir=video_id,
                            ext=transcript_ext,
                            in_memory=in_memory,
                        )

                        a_output_file, audio_arr = utils.trim_audio(
                            audio_file=audio_file,
                            start=start,
                            end=end,
                            output_dir=video_id,
                            in_memory=in_memory,
                        )

                    if audio_arr is not None and not utils.too_short_audio(
                        audio_arr=audio_arr
                    ):
                        segments_list.append(
                            (t_output_file, transcript_string, a_output_file, audio_arr)
                        )
                        segment_count += 1
                    else:
                        if audio_arr is None:
                            with open(f"{log_dir}/faulty_audio.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                a = b

            if b == len(transcript) and diff < 30000:
                over_ctx_len, err = utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript, language=language
                )
                if not over_ctx_len:
                    t_output_file, transcript_string = utils.write_segment(
                        timestamps=timestamps[a:b],
                        transcript=transcript,
                        output_dir=video_id,
                        ext=transcript_ext,
                        in_memory=in_memory,
                    )

                    a_output_file, audio_arr = utils.trim_audio(
                        audio_file=audio_file,
                        start=timestamps[a][0],
                        end=timestamps[b - 1][1],
                        output_dir=video_id,
                        in_memory=in_memory,
                    )

                    if audio_arr is not None and not utils.too_short_audio(
                        audio_arr=audio_arr
                    ):
                        segments_list.append(
                            (t_output_file, transcript_string, a_output_file, audio_arr)
                        )
                        segment_count += 1
                    else:
                        if audio_arr is None:
                            with open(f"{log_dir}/faulty_audio.txt", "a") as f:
                                f.write(f"{video_id}\tindex: {b}\n")
                else:
                    if err is not None:
                        with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")
                    else:
                        with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")

                break
        if len(segments_list) == 0:
            return None
        return segments_list
    except ValueError as e:
        with open(f"{log_dir}/failed_chunking.txt", "a") as f:
            f.write(f"{video_id}\t{e}\n")
        return None
    except Exception as e:
        with open(f"{log_dir}/failed_chunking.txt", "a") as f:
            f.write(f"{video_id}\t{e}\n")
        return None


def parallel_chunk_audio_transcript(
    args,
) -> Optional[List[Tuple[str, str, str, np.ndarray]]]:
    """Parallelized version of chunk_audio_transcript function to work in multiprocessing context"""
    return chunk_audio_transcript(*args)


def chunk_transcript(
    transcript_data: Dict,
    transcript_manifest: List[str],
    log_dir: str,
    in_memory: bool = True,
) -> Optional[List[Tuple[str, str, str, np.ndarray]]]:
    """Segment audio and transcript files into <= 30-second chunks

    Segment audio and transcript files into <= 30-second chunks. The audio and transcript files are represented by audio_file and transcript_file respectively.

    Args:
    transcript_file: Path to the transcript file
    audio_file: Path to the audio file

    Raises:
        Exception: If an error occurs during the chunking process
    """
    try:
        transcript_string = transcript_data["content"]
        transcript_file = transcript_data["subtitle_file"]
        print(f"{transcript_file=}")
        if transcript_file.startswith("/weka"):
            video_id = transcript_file.split("/")[6]
        else:
            video_id = transcript_file.split("/")[1]
        print(f"{video_id=}")

        transcript_ext = transcript_file.split(".")[-1]
        segment_count = 0

        transcript, *_ = utils.TranscriptReader(
            file_path=None, transcript_string=transcript_string, ext=transcript_ext
        ).read()

        if len(transcript.keys()) == 0:
            with open(f"{log_dir}/empty_transcripts.txt", "a") as f:
                f.write(f"{video_id}\n")
            return None

        a = 0
        b = 0

        timestamps = list(transcript.keys())
        diff = 0
        init_diff = 0
        segments_list = []

        while a < len(transcript) + 1 and segment_count < SEGMENT_COUNT_THRESHOLD:
            init_diff = utils.calculate_difference(timestamps[a][0], timestamps[b][1])

            if init_diff < 30000:
                diff = init_diff
                b += 1
            else:
                # edge case (when transcript line is > 30s)
                if b == a:
                    with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                        f.write(f"{video_id}\tindex: {b}\n")

                    a += 1
                    b += 1

                    if a == b == len(transcript):
                        if segment_count == 0:
                            with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                                f.write(f"{video_id}\tdelete\n")
                        break

                    continue

                over_ctx_len, err = utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript, language=None
                )
                if not over_ctx_len:
                    t_output_file, transcript_string = utils.write_segment(
                        timestamps=timestamps[a:b],
                        transcript=transcript,
                        output_dir=video_id,
                        ext=transcript_ext,
                        in_memory=in_memory,
                    )

                    if not utils.too_short_audio_text(
                        start=timestamps[a][0], end=timestamps[b - 1][1]
                    ):
                        segments_list.append((t_output_file, transcript_string))
                        segment_count += 1
                else:
                    if err is not None:
                        with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")
                    else:
                        with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")

                init_diff = 0
                diff = 0

                # checking for silence
                if timestamps[b][0] > timestamps[b - 1][1]:
                    silence_segments = (
                        utils.calculate_difference(
                            timestamps[b - 1][1], timestamps[b][0]
                        )
                        // 30000
                    )

                    for i in range(0, silence_segments + 1):
                        start = utils.adjust_timestamp(
                            timestamps[b - 1][1], (i * 30000)
                        )

                        if i == silence_segments:
                            if start == timestamps[b][0]:
                                continue
                            else:
                                end = timestamps[b][0]
                        else:
                            end = utils.adjust_timestamp(start, 30000)

                        t_output_file, transcript_string = utils.write_segment(
                            timestamps=[(start, end)],
                            transcript=None,
                            output_dir=video_id,
                            ext=transcript_ext,
                            in_memory=in_memory,
                        )

                    if not utils.too_short_audio_text(start=start, end=end):
                        segments_list.append((t_output_file, transcript_string))
                        segment_count += 1
                a = b

            if b == len(transcript) and diff < 30000:
                over_ctx_len, err = utils.over_ctx_len(
                    timestamps=timestamps[a:b], transcript=transcript, language=None
                )
                if not over_ctx_len:
                    t_output_file, transcript_string = utils.write_segment(
                        timestamps=timestamps[a:b],
                        transcript=transcript,
                        output_dir=video_id,
                        ext=transcript_ext,
                        in_memory=in_memory,
                    )

                    if not utils.too_short_audio_text(
                        start=timestamps[a][0], end=timestamps[b - 1][1]
                    ):
                        segments_list.append((t_output_file, transcript_string))
                        segment_count += 1
                else:
                    if err is not None:
                        with open(f"{log_dir}/faulty_transcripts.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")
                    else:
                        with open(f"{log_dir}/over_ctx_len.txt", "a") as f:
                            f.write(f"{video_id}\tindex: {b}\n")

                break
        if len(segments_list) == 0:
            return None

        segments_list = [
            segment for segment in segments_list if segment[0] in transcript_manifest
        ]
        return segments_list
    except ValueError as e:
        with open(f"{log_dir}/failed_chunking.txt", "a") as f:
            f.write(f"{video_id}\t{e}\n")
        return None
    except Exception as e:
        with open(f"{log_dir}/failed_chunking.txt", "a") as f:
            f.write(f"{video_id}\t{e}\n")
        return None


def parallel_chunk_transcript(
    args,
) -> Optional[List[Tuple[str, str, str, np.ndarray]]]:
    """Parallelized version of chunk_audio_transcript function to work in multiprocessing context"""
    return chunk_transcript(*args)


def get_missing_pairs(audio_files, transcript_files):
    audio_file_names = [path.split("/")[1] for path in audio_files]
    transcript_file_names = [path.split("/")[1] for path in transcript_files]

    if len(audio_files) > len(transcript_files):
        missing_pairs = [
            path
            for path in audio_files
            if path.split("/")[1] not in transcript_file_names
        ]
        audio_files = sorted(list(set(audio_files) - set(missing_pairs)))
    elif len(audio_files) < len(transcript_files):
        missing_pairs = [
            path
            for path in transcript_files
            if path.split("/")[1] not in audio_file_names
        ]
        transcript_files = sorted(list(set(transcript_files) - set(missing_pairs)))

    return audio_files, transcript_files, missing_pairs


def preprocess(
    tar_file: str,
    log_dir: str,
    seg_dir: str,
    audio_dir: str,
    in_memory: bool = True,
) -> None:
    shard = tar_file.split(".")[0]
    log_dir = os.path.join(log_dir, shard)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    transcript_files = list_files_with_extension(tar_file, TRANSCRIPT_EXT)
    audio_files = list_files_with_extension(tar_file, AUDIO_EXT)

    logger.info(f"{len(audio_files)} audio files")
    logger.info(f"{len(transcript_files)} transcript files")

    if len(audio_files) != len(transcript_files):
        logger.info(f"Uneven number of audio and transcript files in {tar_file}")
        audio_files, transcript_files, missing_pairs = get_missing_pairs(
            audio_files, transcript_files
        )

        logger.info(f"{len(audio_files)} audio files")
        logger.info(f"{len(transcript_files)} transcript files")

        with open(
            f"{log_dir}/missing_pairs.txt",
            "a",
        ) as f:
            [f.write(f"{path.split('/')[1]}\n") for path in missing_pairs]

    with multiprocessing.Pool() as pool:
        audio_files = sorted(
            list(
                tqdm(
                    pool.imap_unordered(
                        parallel_read_file_in_tar,
                        zip(repeat(tar_file), audio_files, repeat(audio_dir)),
                    ),
                    total=len(audio_files),
                )
            )
        )

    print([p for p in audio_files[-5:]])
    print([p for p in transcript_files[-5:]])

    # Chunk the audio and transcript files
    logger.info("Chunking audio and transcript files")
    start = time.time()
    with multiprocessing.Pool(multiprocessing.cpu_count() * 7) as pool:
        segments_group = list(
            tqdm(
                pool.imap_unordered(
                    parallel_chunk_audio_transcript,
                    zip(
                        repeat(tar_file),
                        transcript_files,
                        audio_files,
                        repeat(log_dir),
                        repeat(in_memory),
                    ),
                ),
                total=len(transcript_files),
            )
        )
    logger.info(segments_group[:5])
    # segments group is [[(t_output_file, transcript_string, a_output_file, audio_arr), ...], ...]
    # where each inner list is a group of segments from one audio-transcript file, and each tuple is a segment
    segments_group = [group for group in segments_group if group is not None]
    logger.info(f"Time taken to segment: {(time.time() - start) / 60} minutes")

    # Write the data to tar files
    print("Writing data to tar files")
    start = time.time()
    segments_list = list(chain(*segments_group))
    seg_tar_path = write_to_tar(shard, segments_list, seg_dir)
    print(f"Time taken to write to tar files: {time.time() - start}")

    return seg_tar_path, log_dir


def preprocess_jsonl(
    json_file: str,
    transcript_manifest_file: str,
    log_dir: str,
    seg_dir: str,
    in_memory: bool = True,
):
    shard = (
        json_file.split(".")[0].split("_")[-1]
        if json_file.startswith("shard")
        else json_file.split(".")[0]
    )
    log_dir = os.path.join(log_dir, shard)
    os.makedirs(log_dir, exist_ok=True)

    if json_file.endswith(".gz"):
        transcript_data = unarchive_jsonl_gz(json_file)
    else:
        with open(json_file, "r") as f:
            transcript_data = [json.loads(line.strip()) for line in f]

    with open(transcript_manifest_file, "r") as f:
        transcript_manifest = [line.strip() for line in f]

    logger.info("Chunking audio and transcript files")
    start = time.time()
    with multiprocessing.Pool() as pool:
        segments_group = list(
            tqdm(
                pool.imap_unordered(
                    parallel_chunk_transcript,
                    zip(
                        transcript_data,
                        repeat(transcript_manifest),
                        repeat(log_dir),
                        repeat(in_memory),
                    ),
                ),
                total=len(transcript_data),
            )
        )

    logger.info(segments_group[:5])
    segments_group = [group for group in segments_group if group is not None]
    logger.info(f"{segments_group[:5]=}")
    logger.info(f"Time taken to segment: {(time.time() - start) / 60} minutes")

    # Write the data to tar files
    logger.info("Writing data to tar files")
    start = time.time()
    segments_list = list(chain(*segments_group))
    logger.info(f"{segments_list[:5]=}")
    seg_tar_path = write_to_tar_text(shard, segments_list, seg_dir)
    logger.info(f"Time taken to write to tar files: {time.time() - start}")

    return seg_tar_path, log_dir


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads data to a GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    logger.info(f"File {destination_blob_name} uploaded to {bucket_name}.")


def upload_dir_to_gcs(local_dir, bucket, prefix):
    cmd = ["gsutil", "-m", "cp", "-r", f"{local_dir}", f"gs://{bucket}/{prefix}/"]

    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode != 0:
        logger.error(
            f"Error: Command failed with return code {res.returncode}, {res.stderr}"
        )
        return None


def compress_dir(source_dir, tar_file):
    with tarfile.open(tar_file, "w:gz") as tar:
        tar.add(source_dir, arcname=".")


def get_vm_info():
    cmd = [
        "curl",
        "http://metadata.google.internal/computeMetadata/v1/instance/name",
        "-H",
        "Metadata-Flavor: Google",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    instance_name = result.stdout.strip()

    cmd = [
        "curl",
        "http://metadata.google.internal/computeMetadata/v1/instance/zone",
        "-H",
        "Metadata-Flavor: Google",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    zone = result.stdout.strip().split("/")[-1]

    return instance_name, zone


def shutdown_vm(project, zone, instance_name):
    client = compute_v1.InstancesClient()

    # Request to stop the instance
    operation = client.delete(project=project, zone=zone, instance=instance_name)

    # Wait for the operation to complete
    operation.result()


def process_tar_file(bucket, item, tar_prefix, log_dir, seg_dir, audio_dir):
    gcs_tar_file = os.path.join(tar_prefix, item["tarFile"])
    tar_file = item["tarFile"]

    logger.info(f"Downloading {gcs_tar_file} from GCS..")
    download_from_gcs(bucket, gcs_tar_file, tar_file)

    seg_tar_path, seg_log_path = preprocess(
        tar_file=tar_file,
        log_dir=log_dir,
        seg_dir=seg_dir,
        audio_dir=audio_dir,
        in_memory=True,
    )

    logger.info(f"Uploading {seg_tar_path} to GCS..")
    upload_to_gcs(
        bucket_name=bucket,
        source_file_name=seg_tar_path,
        destination_blob_name=seg_tar_path,
    )

    logger.info(f"Uploading metadata to GCS..")
    upload_dir_to_gcs(seg_log_path, bucket, log_dir)

    # Removing files for next iteration
    logger.info("Cleaning up..\n")
    os.remove(tar_file)
    os.remove(seg_tar_path)
    shutil.rmtree(log_dir)
    shutil.rmtree(audio_dir)


def process_jsonl_file(bucket, item, jsonl_prefix, manifest_prefix, log_dir, seg_dir):
    gcs_jsonl_file = os.path.join(jsonl_prefix, item["jsonlFile"])
    jsonl_file = item["jsonlFile"]

    logger.info(f"Downloading {gcs_jsonl_file} from GCS..")
    download_from_gcs(bucket, gcs_jsonl_file, jsonl_file)
    transcript_manifest_file = jsonl_file.split("_")[-1].split(".")[0] + ".txt"
    gcs_transcript_manifest_file = os.path.join(manifest_prefix, transcript_manifest_file)
    logger.info(f"Downloading {gcs_transcript_manifest_file} from GCS..")
    download_from_gcs(bucket, gcs_transcript_manifest_file, transcript_manifest_file)

    seg_tar_path, seg_log_path = preprocess_jsonl(
        json_file=jsonl_file,
        transcript_manifest_file=transcript_manifest_file,
        log_dir=log_dir,
        seg_dir=seg_dir,
        in_memory=True,
    )

    logger.info(f"Uploading {seg_tar_path} to GCS..")
    upload_to_gcs(
        bucket_name=bucket,
        source_file_name=seg_tar_path,
        destination_blob_name=seg_tar_path,
    )

    logger.info(f"Uploading metadata to GCS..")
    upload_dir_to_gcs(seg_log_path, bucket, log_dir)

    # Removing files for next iteration
    logger.info("Cleaning up..\n")
    os.remove(jsonl_file)
    os.remove(seg_tar_path)
    shutil.rmtree(log_dir)


def main(
    bucket: str,
    queue_id: str,
    log_dir: str,
    seg_dir: str,
    transcript_only: bool,
    audio_dir: Optional[str] = None,
    tar_prefix: Optional[str] = None,
    jsonl_prefix: Optional[str] = None,
    manifest_prefix: Optional[str] = None,
):
    instance_name, zone = get_vm_info()
    logger.info(f"Instance name: {instance_name}..\n")
    qm = QueueManager(queue_id)

    if len(glob.glob("*.tar.gz")) > 0:
        [os.remove(f) for f in glob.glob("*.tar.gz")]
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    if os.path.exists(seg_dir):
        shutil.rmtree(seg_dir)
    if audio_dir is not None:
        if os.path.exists(audio_dir):
            shutil.rmtree(audio_dir)

    while True:
        try:
            logger.info("Getting next item from queue..")
            message, item = qm.get_next(
                visibility_timeout=(60 * 60)
            )  # 1 hour visibility timeout

            if not transcript_only:
                process_tar_file(
                    bucket=bucket,
                    item=item,
                    tar_prefix=tar_prefix,
                    log_dir=log_dir,
                    seg_dir=seg_dir,
                    audio_dir=audio_dir,
                )
            else:
                process_jsonl_file(
                    bucket=bucket,
                    item=item,
                    jsonl_prefix=jsonl_prefix,
                    manifest_prefix=manifest_prefix,
                    log_dir=log_dir,
                    seg_dir=seg_dir,
                )
            qm.delete(message)
        except IndexError:  # no more items in queue to process
            # break
            logger.info("No more items in queue to process")
            shutdown_vm("oe-training", zone, instance_name)
            break


if __name__ == "__main__":
    Fire(main)
