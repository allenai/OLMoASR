import ray
from ray.data.datasource import FilenameProvider
import glob
import os
from typing import Tuple, Union, Dict, Any, Literal
from open_whisper.utils import TranscriptReader
from fire import Fire
import numpy as np
import io


class FilenameProviders:
    @staticmethod
    class LabelsDictsFilenameProvider(FilenameProvider):
        def __init__(self, file_format: str):
            self.file_format = file_format

        def get_filename_for_block(self, block, task_index, block_index):
            return f"labels_dicts.{self.file_format}"

    @staticmethod
    class SamplesDictsFilenameProvider(FilenameProvider):
        def __init__(self, file_format: str):
            self.file_format = file_format

        def get_filename_for_block(self, block, task_index, block_index):
            return f"samples_dicts.{self.file_format}"


class DataReader:
    @staticmethod
    def bytes_to_text(data_dict: Dict[str, Any]) -> Dict[str, Any]:
        data_dict["text"] = data_dict["bytes"].decode("utf-8")
        del data_dict["bytes"]
        return data_dict

    @staticmethod
    def bytes_to_array(data_dict: Dict[str, Any]) -> Dict[str, Any]:
        data_dict["array"] = np.load(io.BytesIO(data_dict["bytes"]))
        del data_dict["bytes"]
        return data_dict


class DataLabeler:
    @staticmethod
    def check_case(transcript_dict: Dict[str, Any]) -> Dict[str, Any]:
        reader = TranscriptReader(transcript_string=transcript_dict["text"], ext="srt")
        t_dict, *_ = reader.read()
        text = reader.extract_text(t_dict)

        seg_dir_label_dict = {}
        seg_dir_label_dict["seg_dir"] = os.path.dirname(
            transcript_dict["path"]
        ).replace("440K_full", "440K_seg")

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


def gen_smpl_dict(segs_dir: Dict[str, Any]) -> Dict[str, Any]:
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


class DataFilter:
    def __init__(
        self,
        data_dir: str,
        label_data_dir: str,
        samples_dicts_dir: str,
        labels: str,
        batch_size: int,
        batch_idx: int,
        filter_mode: bool,
        generate_mode: str,
    ):
        self.data_dir = data_dir
        self.label_data_dir = label_data_dir
        self.samples_dicts_dir = samples_dicts_dir
        self.labels = labels
        self.batch_size = batch_size
        self.batch_idx = batch_idx
        self.filter_mode = filter_mode
        self.generate_mode = generate_mode

    def remove_lower(self):
        ray.init(num_cpus=20, num_gpus=1)
        # Loading data and converting binary data to text
        if self.generate_mode:
            data_dirs = [
                f"{self.data_dir}/{((self.batch_idx * self.batch_size) + i):05}"
                for i in range(self.batch_size)
                if (self.batch_idx * self.batch_size) + i <= 2448
            ]
            print(data_dirs[:5])

            print("Start reading binary files")
            ds = ray.data.read_binary_files(
                paths=data_dirs, file_extensions=["srt"], include_paths=True
            ).map(DataReader.bytes_to_text)

            # Inspecting to execute transformation
            print("Finish reading binary files")
            print(ds.count())

            # Labeling the data
            ds = ds.map(DataLabeler.check_case)

            # Repartitioning the data to 1 block to write to 1 file only
            ds.repartition(num_blocks=1).write_json(
                self.label_data_dir,
                filename_provider=FilenameProviders.LabelsDictsFilenameProvider(
                    "jsonl"
                ),
            )
            print(f"Finish writing the labeled data to {self.label_data_dir}")

            # Inspecting to execute transformation
            print("Finish labeling the data")
            print(ds.count())
        elif self.filter_mode:
            ds = ray.data.read_json(self.label_data_dir, file_format="jsonl")
            print(ds.count())

        print("Filtering labeled data")
        ds = ds.map(filter_in_label, fn_kwargs={"labels": self.labels})
        print(ds.count())
        print("Finish filtering labeled data")

        ds = ds.map(gen_smpl_dict)
        ds.repartition(num_blocks=1).write_json(
            self.samples_dicts_dir,
            filename_provider=FilenameProviders.SamplesDictsFilenameProvider("jsonl"),
        )
        print(
            f"Finish generating samples dicts to {self.samples_dicts_dir}/samples_dicts.jsonl"
        )

if __name__ == "__main__":
    Fire(DataFilter)
    
    
    
    
    
    
    
    
    
    
    
    
# ray.init(num_cpus=20, num_gpus=1)
# if generate_mode:
#     data_dirs = [
#         f"{data_dir}/{((batch_idx * batch_size) + i):05}"
#         for i in range(batch_size)
#         if (batch_idx * batch_size) + i <= 2448
#     ]
#     print(data_dirs[:5])

#     # read in data
#     ds = ray.data.read_binary_files(
#         paths=data_dirs, file_extensions=[], include_paths=True
#     ).map(DataReader.bytes_to_text)

#     # label data

#     # Repartitioning the data to 1 block to write to 1 file only
#     ds.repartition(num_blocks=1).write_json(
#         label_data_dir, filename_provider=FilenameProviders.LabelsDictsFilenameProvider("jsonl")
#     )
#     print(f"Finish writing the labeled data to {label_data_dir}")
# elif filter_mode:
#     ds = ray.data.read_json(label_data_dir, file_format="jsonl")
#     print(ds.count())

# print("Filtering labeled data")
# ds = ds.map(filter_in_label, fn_kwargs={"labels": labels})
# print(ds.count())
# print("Finish filtering labeled data")

# ds = ds.map(gen_smpl_dict)
# ds.repartition(num_blocks=1).write_json(
#     samples_dicts_dir, filename_provider=FilenameProviders.SamplesDictsFilenameProvider("jsonl")
# )
# print(f"Finish generating samples dicts to {samples_dicts_dir}/samples_dicts.jsonl")
