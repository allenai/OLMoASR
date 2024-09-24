import ray
from ray.data.datasource import FilenameProvider
import glob
import os
from typing import Tuple, Union, Dict, Any, Literal
from open_whisper.utils import TranscriptReader
from fire import Fire


class LabelsDictsFilenameProvider(FilenameProvider):
    def __init__(self, file_format: str):
        self.file_format = file_format

    def get_filename_for_block(self, block, task_index, block_index):
        return f"labels_dicts.{self.file_format}"


class SamplesDictsFilenameProvider(FilenameProvider):
    def __init__(self, file_format: str):
        self.file_format = file_format

    def get_filename_for_block(self, block, task_index, block_index):
        return f"samples_dicts.{self.file_format}"


def bytes_to_text(transcript_dict: Dict[str, Any]) -> Dict[str, Any]:
    transcript_dict["text"] = transcript_dict["bytes"].decode("utf-8")
    del transcript_dict["bytes"]
    return transcript_dict


def check_case(transcript_dict: Dict[str, Any]) -> Dict[str, Any]:
    reader = TranscriptReader(transcript_string=transcript_dict["text"], ext="srt")
    t_dict, *_ = reader.read()
    text = reader.extract_text(t_dict)

    seg_dir_label_dict = {}
    seg_dir_label_dict["seg_dir"] = os.path.dirname(transcript_dict["path"]).replace(
        "440K_full", "440K_seg"
    )

    if text.islower():
        seg_dir_label_dict["label"] = "LOWER"
    elif text.isupper():
        seg_dir_label_dict["label"] = "UPPER"
    elif text == "":
        seg_dir_label_dict["label"] = "EMPTY"
    else:
        seg_dir_label_dict["label"] = "MIXED"

    return seg_dir_label_dict


def filter_in_label(
    label_dict: Dict[str, Any], labels: Union[str, Tuple[str]]
) -> Dict[str, Any]:
    if isinstance(labels, str):
        if label_dict["label"] == labels:
            return {"seg_dir": label_dict["seg_dir"]}
        else:
            return {"seg_dir": None}
    elif isinstance(labels, Tuple):
        if label_dict["label"] in labels:
            return {"seg_dir": label_dict["seg_dir"]}
        else:
            return {"seg_dir": None}
    else:
        return {"seg_dir": None}


def gen_smpl_dict(segs_dir) -> Dict[str, Any]:
    segs_dir = segs_dir["seg_dir"]
    if segs_dir is not None:
        srt_files = sorted(glob.glob(segs_dir + "/*.srt"))
        npy_files = sorted(glob.glob(segs_dir + "/*.npy"))
        srt_npy_samples = list(zip(srt_files, npy_files))
        smpl_dicts = []

        for srt_fp, npy_fp in srt_npy_samples:
            smpl_dict = {"key": segs_dir, "srt": srt_fp, "npy": npy_fp}
            smpl_dicts.append(smpl_dict)

        return {"sample_dicts": smpl_dicts}
    else:
        return {"sample_dicts": None}


def remove_lower(
    data_dir: str,
    label_data_dir: str,
    samples_dicts_dir: str,
    labels: str,
    batch_size: int,
    batch_idx: int,
    filter_mode: bool,
    generate_mode: bool,
) -> None:
    ray.init(num_cpus=20, num_gpus=1)
    # Loading data and converting binary data to text
    if generate_mode:
        data_dirs = [
            f"{data_dir}/{((batch_idx * batch_size) + i):05}"
            for i in range(batch_size)
            if (batch_idx * batch_size) + i <= 2448
        ]
        print(data_dirs[:5])

        print("Start reading binary files")
        ds = ray.data.read_binary_files(
            paths=data_dirs, file_extensions=["srt"], include_paths=True
        ).map(bytes_to_text)

        # Inspecting to execute transformation
        print("Finish reading binary files")
        print(ds.count())

        # Labeling the data
        ds = ds.map(check_case)

        # Repartitioning the data to 1 block to write to 1 file only
        ds.repartition(num_blocks=1).write_json(
            label_data_dir, filename_provider=LabelsDictsFilenameProvider("jsonl")
        )
        print(f"Finish writing the labeled data to {label_data_dir}")

        # Inspecting to execute transformation
        print("Finish labeling the data")
        print(ds.count())
    elif filter_mode:
        ds = ray.data.read_json(label_data_dir, file_format="jsonl")
        print(ds.count())

    print("Filtering labeled data")
    ds = ds.map(filter_in_label, fn_kwargs={"labels": labels})
    print(ds.count())
    print("Finish filtering labeled data")

    ds = ds.map(gen_smpl_dict)
    ds.repartition(num_blocks=1).write_json(
        samples_dicts_dir, filename_provider=SamplesDictsFilenameProvider("jsonl")
    )
    print(f"Finish generating samples dicts to {samples_dicts_dir}/samples_dicts.jsonl")


if __name__ == "__main__":
    Fire(remove_lower)

# # Loading the labeled data
# with open(os.path.join(label_data_dir, "labels_dicts.jsonl"), "r") as f:
#     label_dicts = [json.loads(line) for line in f]

# print(f"Label dicts: {label_dicts[:2]}")
# segs_dirs = mp_filter_in_label(label_dicts=label_dicts, labels=labels)
# print(f"Segs dirs: {segs_dirs[:2]}")

# ds = ray.data.from_items(segs_dirs)
