from vllm import LLM
import os
from tqdm import tqdm
import torch
import json
import gzip
import glob
import multiprocessing
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict

def open_file(file_path) -> List[Dict]:
    with gzip.open(file_path, "rt") as f:
        data = [json.loads(line.strip()) for line in f]
    return file_path, data


class SamplesDictsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        man_seg_text = self.data[idx]["seg_text"]
        mach_seg_text = self.data[idx]["mach_seg_text"]
        return man_seg_text, mach_seg_text


def process_jsonl(llm, data, batch_size, num_workers):
    dataset = SamplesDictsDataset(data)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    device = torch.device("cuda")
    scores = []
    for _, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        man_seg_text, mach_seg_text = batch
        man_seg_text = man_seg_text.to(device)
        mach_seg_text = mach_seg_text.to(device)

        output = llm.score(man_seg_text, mach_seg_text)
        score = output.outputs.score
        scores.extend(score)

    for i, segment in enumerate(len(data)):
        segment["seg_man_mach_score"] = scores[i]

    return data


def main(
    source_dir,
    output_dir,
    start_shard_idx: int,
    job_batch_size: int,
    batch_size: int,
    num_workers: int,
):
    os.makedirs(output_dir, exist_ok=True)
    job_idx = int(os.getenv("BEAKER_REPLICA_RANK"))
    job_start_shard_idx = start_shard_idx + (job_idx * job_batch_size)
    job_end_shard_idx = start_shard_idx + ((job_idx + 1) * job_batch_size)
    print(f"{job_start_shard_idx=}")
    print(f"{job_end_shard_idx=}")
    data_shard_paths = sorted(glob.glob(source_dir + "/*.jsonl.gz"))[
        job_start_shard_idx : job_end_shard_idx
        + 1  # forgot to add 1 here! so last shard in output file is excluded
    ]
    print(f"{len(data_shard_paths)=}")
    print(f"{data_shard_paths[:5]=}")
    print(f"{data_shard_paths[-5:]=}")

    with multiprocessing.Pool() as pool:
        result = list(
            tqdm(
                pool.imap_unordered(open_file, data_shard_paths),
                total=len(data_shard_paths),
            )
        )

    shard_paths, all_data = zip(*result)
    print(f"All segments in job: {sum(len(data) for data in all_data)}")
    
    llm = LLM(model="Qwen/Qwen2-1.5B", task="score")

    new_all_data = [
        process_jsonl(llm, data, batch_size, num_workers) for data in all_data
    ]

    for shard_path, data in zip(shard_paths, new_all_data):
        with gzip.open(f"{output_dir}/{os.path.basename(shard_path)}", "wt") as f:
            for seg in data:
                f.write(json.dumps(seg) + "\n")
