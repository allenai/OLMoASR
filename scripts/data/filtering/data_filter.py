import ray
from ray.data.datasource import FilenameProvider
import glob
import os
from typing import Tuple, Union, Dict, Any, Literal, Optional
from open_whisper.utils import TranscriptReader
from fire import Fire
import numpy as np
import io
from collections import defaultdict


class FilenameProviders:
    def __init__(self):
        pass
    
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
    def __init__(self):
        pass

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


class FilterFunc:
    def __init__(self):
        pass

    @staticmethod
    def not_lower(row: Dict[str, Any]) -> bool:
        return not row["text"].islower()

    @staticmethod
    def not_lower_empty(row: Dict[str, Any]) -> bool:
        if not (row["text"].islower() or row["text"].strip() == ""):
            return True
        else:
            return False
    
    @staticmethod
    def not_upper(row: Dict[str, Any]) -> bool:
        return not row["text"].isupper()

    @staticmethod
    def not_upper_empty(row: Dict[str, Any]) -> bool:
        if not (row["text"].isupper() or row["text"].strip() == ""):
            return True
        else:
            return False
    
    @staticmethod
    def not_lower_upper(row: Dict[str, Any]) -> bool:
        return not (row["text"].islower() or row["text"].isupper())
    
    @staticmethod
    def only_mixed(row: Dict[str, Any]) -> bool:
        if not(row["text"].islower() or row["text"].isupper() or row["text"].strip() == ""):
            return True
        else:
            return False

    @staticmethod
    def min_comma_period(row: Dict[str, Any]) -> bool:
        if "," in row["text"] and "." in row["text"]:
            return True
        else:
            return False

    @staticmethod
    def min_comma_period_exclaim(row: Dict[str, Any]) -> bool:
        if "," in row["text"] and "." in row["text"] and "!" in row["text"]:
            return True
        else:
            return False

    @staticmethod
    def min_comma_period_question(row: Dict[str, Any]) -> bool:
        if "," in row["text"] and "." in row["text"] and "?" in row["text"]:
            return True
        else:
            return False

    @staticmethod
    def min_comma_period_question_exclaim(row: Dict[str, Any]) -> bool:
        if (
            "," in row["text"]
            and "." in row["text"]
            and "!" in row["text"]
            and "?" in row["text"]
        ):
            return True
        else:
            return False

    @staticmethod
    def no_repeat(row: Dict[str, Any]):
        reader = TranscriptReader(
            file_path=None, transcript_string=row["text"], ext="srt"
        )
        t_dict, *_ = reader.read()
        transcript_text_list = list(t_dict.values())
        unique_text = set(transcript_text_list)

        if len(transcript_text_list) != len(unique_text):
            return False
        else:
            return True
        
    @staticmethod
    def no_upper_no_repeat(row: Dict[str, Any]):
        if not row["text"].isupper() and FilterFunc.no_repeat(row):
            return True
        else:
            return False

    @staticmethod
    def no_lower_no_repeat(row: Dict[str, Any]):
        if not row["text"].islower() and FilterFunc.no_repeat(row):
            return True
        else:
            return False
        
    @staticmethod
    def min_comma_period_no_repeat(row: Dict[str, Any]):
        if FilterFunc.min_comma_period(row) and FilterFunc.no_repeat(row):
            return True
        else:
            return False

    @staticmethod
    def mixed_no_repeat(row: Dict[str, Any]):
        if FilterFunc.only_mixed(row) and FilterFunc.no_repeat(row):
            return True
        else:
            return False

def gen_smpl_dict(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    segs_dir = os.path.dirname(row["path"]).replace("440K_full", "440K_seg")

    srt_files = sorted(glob.glob(segs_dir + "/*.srt"))
    npy_files = sorted(glob.glob(segs_dir + "/*.npy"))
    srt_npy_samples = list(zip(srt_files, npy_files))
    smpl_dicts = []

    for srt_fp, npy_fp in srt_npy_samples:
        smpl_dict = {"key": segs_dir, "srt": srt_fp, "npy": npy_fp}
        smpl_dicts.append(smpl_dict)

    row["sample_dicts"] = smpl_dicts
    del row["path"]
    del row["text"]

    return row


class DataFilter:
    def __init__(
        self,
        data_dir: str,
        samples_dicts_dir: str,
        batch_size: int,
        filter_mode: bool,
        metadata_path: str,
    ):
        self.data_dir = data_dir
        self.samples_dicts_dir = samples_dicts_dir + f"/{int(os.getenv("BEAKER_REPLICA_INDEX")):03}"
        self.batch_size = batch_size
        self.batch_idx = int(os.getenv("BEAKER_REPLICA_INDEX"))
        self.filter_mode = filter_mode
        self.metadata_path = metadata_path
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

    def base_filter(self, filter_func: FilterFunc):
        ray.init(num_cpus=20, num_gpus=0)
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

        print("Finish reading binary files")
        total = ds.count()

        if not self.filter_mode:
            ds = ds.filter(filter_func)
            filtered = ds.count()
            removed = total - filtered

            return (removed, total)

        ds = ds.filter(filter_func).map(gen_smpl_dict)
        filtered = ds.count()
        removed = total - filtered
        ds.repartition(num_blocks=1).write_json(
            self.samples_dicts_dir,
            filename_provider=FilenameProviders.SamplesDictsFilenameProvider("jsonl"),
        )
        return (removed, total)

    def not_lower(self):
        removed_count, total_count = self.base_filter(filter_func=FilterFunc.not_lower)

        with open(self.metadata_path, "a") as f:
            f.write(f"Removed {removed_count} out of {total_count} samples\n")

    def not_lower_empty(self):
        removed_count, total_count = self.base_filter(
            filter_func=FilterFunc.not_lower_empty
        )

        with open(self.metadata_path, "a") as f:
            f.write(f"Removed {removed_count} out of {total_count} samples\n")
            
    def not_upper(self):
        removed_count, total_count = self.base_filter(filter_func=FilterFunc.not_upper)

        with open(self.metadata_path, "a") as f:
            f.write(f"Removed {removed_count} out of {total_count} samples\n")
        
    def not_upper_empty(self):
        removed_count, total_count = self.base_filter(
            filter_func=FilterFunc.not_upper_empty
        )

        with open(self.metadata_path, "a") as f:
            f.write(f"Removed {removed_count} out of {total_count} samples\n")
    
    def not_lower_upper(self):
        removed_count, total_count = self.base_filter(
            filter_func=FilterFunc.not_lower_upper
        )

        with open(self.metadata_path, "a") as f:
            f.write(f"Removed {removed_count} out of {total_count} samples\n")
    
    def only_mixed(self):
        removed_count, total_count = self.base_filter(
            filter_func=FilterFunc.only_mixed
        )

        with open(self.metadata_path, "a") as f:
            f.write(f"Removed {removed_count} out of {total_count} samples\n")

    def min_comma_period(self):
        removed_count, total_count = self.base_filter(
            filter_func=FilterFunc.min_comma_period
        )

        with open(self.metadata_path, "a") as f:
            f.write(f"Removed {removed_count} out of {total_count} samples\n")

    def min_comma_period_exclaim(self):
        removed_count, total_count = self.base_filter(
            filter_func=FilterFunc.min_comma_period_exclaim
        )

        with open(self.metadata_path, "a") as f:
            f.write(f"Removed {removed_count} out of {total_count} samples\n")

    def min_comma_period_question(self):
        removed_count, total_count = self.base_filter(
            filter_func=FilterFunc.min_comma_period_question
        )

        with open(self.metadata_path, "a") as f:
            f.write(f"Removed {removed_count} out of {total_count} samples\n")

    def min_comma_period_question_exclaim(self):
        removed_count, total_count = self.base_filter(
            filter_func=FilterFunc.min_comma_period_question_exclaim
        )

        with open(self.metadata_path, "a") as f:
            f.write(f"Removed {removed_count} out of {total_count} samples\n")

    def no_repeat(self):
        removed_count, total_count = self.base_filter(filter_func=FilterFunc.no_repeat)

        with open(self.metadata_path, "a") as f:
            f.write(f"Removed {removed_count} out of {total_count} samples\n")
    
    def no_upper_no_repeat(self):
        removed_count, total_count = self.base_filter(
            filter_func=FilterFunc.no_upper_no_repeat
        )

        with open(self.metadata_path, "a") as f:
            f.write(f"Removed {removed_count} out of {total_count} samples\n")

    def no_lower_no_repeat(self):
        removed_count, total_count = self.base_filter(
            filter_func=FilterFunc.no_lower_no_repeat
        )

        with open(self.metadata_path, "a") as f:
            f.write(f"Removed {removed_count} out of {total_count} samples\n")
    
    def min_comma_period_no_repeat(self):
        removed_count, total_count = self.base_filter(
            filter_func=FilterFunc.min_comma_period_no_repeat
        )

        with open(self.metadata_path, "a") as f:
            f.write(f"Removed {removed_count} out of {total_count} samples\n")
    
    def mixed_no_repeat(self):
        removed_count, total_count = self.base_filter(
            filter_func=FilterFunc.mixed_no_repeat
        )

        with open(self.metadata_path, "a") as f:
            f.write(f"Removed {removed_count} out of {total_count} samples\n")

if __name__ == "__main__":
    Fire(DataFilter)
