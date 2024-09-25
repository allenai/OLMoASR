import ray
from typing import Tuple
import glob
from fire import Fire


def get_removed_count(labels_dicts_dir: str, labels_removed: str):
    ray.init(num_cpus=20, num_gpus=1)

    ds = ray.data.read_json(glob.glob(labels_dicts_dir + "/*/labels_dicts.jsonl"))

    if isinstance(labels_removed, str):
        filter_func = lambda row: row["label"] == labels_removed
    elif isinstance(labels_removed, Tuple):
        filter_func = lambda row: row["label"] in labels_removed

    removed_count = ds.filter(filter_func).count()
    total_count = ds.count()
    portion_removed = (removed_count / total_count) * 100

    return {
        "removed_count": removed_count,
        "total_count": total_count,
        "portion_removed": portion_removed,
    }


if __name__ == "__main__":
    Fire(get_removed_count)
