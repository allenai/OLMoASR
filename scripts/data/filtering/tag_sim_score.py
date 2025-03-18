from vllm import LLM
import os
from tqdm import tqdm
import torch
import json
from time import time
import gzip
import glob
import multiprocessing
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Literal
from fire import Fire
import torch.nn.functional as F
import numpy as np
import webvtt
from open_whisper.utils import TranscriptReader


def normalize(x, p=2, axis=1, eps=1e-12):
    norm = np.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)  # Avoid division by zero


def cosine_similarity(x1, x2, axis=1, eps=1e-8):
    dot_product = np.sum(x1 * x2, axis=axis)
    norm_x1 = np.linalg.norm(x1, axis=axis)
    norm_x2 = np.linalg.norm(x2, axis=axis)
    return dot_product / np.maximum(norm_x1 * norm_x2, eps)  # Avoid division by zero


def open_file(file_path) -> List[Dict]:
    with gzip.open(file_path, "rt") as f:
        data = [json.loads(line.strip()) for line in f]
    return file_path, data


class SamplesDictsDataset(Dataset):
    def __init__(self, data, level):
        self.data = data
        self.level = level

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        man_seg_text = (
            self.data[idx]["seg_text"]
            if self.level == "seg"
            else self.get_man_text(self.data[idx]["content"])
        )
        mach_seg_text = (
            self.data[idx]["mach_seg_text"]
            if self.level == "seg"
            else self.get_mach_text(self.data[idx]["mach_content"])
        )

        if man_seg_text == "":
            man_seg_text = " "

        if mach_seg_text == "":
            mach_seg_text = " "

        return man_seg_text, mach_seg_text

    def get_man_text(self, man_content):
        reader = TranscriptReader(
            file_path=None,
            transcript_string=man_content,
            ext="vtt" if man_content.startswith("WEBVTT") else "srt",
        )
        t_dict, *_ = reader.read()
        man_text = reader.extract_text(t_dict)
        return man_text

    def get_mach_text(self, mach_content):
        content = webvtt.from_string(mach_content)
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

            mach_text = " ".join([caption.text for caption in modified_content])
        else:
            mach_text = ""
        return mach_text


def process_jsonl(llm, data, batch_size, num_workers, level):
    dataset = SamplesDictsDataset(data, level)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    scores = []
    start = time()
    for _, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        man_seg_text, mach_seg_text = batch
        print(f"{len(man_seg_text)=}, {len(mach_seg_text)=}")
        print(f"{man_seg_text[0]=}")
        print(f"{mach_seg_text[0]=}")

        man_output = llm.embed(man_seg_text)
        mach_output = llm.embed(mach_seg_text)

        man_embeds = F.normalize(
            torch.stack(
                [torch.tensor(output.outputs.embedding) for output in man_output], dim=0
            ),
            p=2,
            dim=-1,
        )
        mach_embeds = F.normalize(
            torch.stack(
                [torch.tensor(output.outputs.embedding) for output in mach_output],
                dim=0,
            ),
            p=2,
            dim=-1,
        )

        print(f"{man_embeds.shape=}, {mach_embeds.shape=}")
        print(f"{man_embeds[0]=}")
        print(f"{mach_embeds[0]=}")

        batch_score = F.cosine_similarity(man_embeds, mach_embeds, dim=-1)
        print(f"{batch_score.shape=}")
        batch_score = batch_score.tolist()

        scores.extend(batch_score)

    print(f"Time taken: {time() - start}")
    print(f"Number of scores: {len(scores)}")
    print(f"Number of segments: {len(data)}")
    assert len(scores) == len(data)

    for i, segment in enumerate(data):
        segment["seg_man_mach_score"] = scores[i]

    return data


def main(
    source_dir,
    output_dir,
    # start_shard_idx: int,
    # job_batch_size: int,
    batch_size: int,
    num_workers: int,
    level: Literal["seg", "doc"],
):
    os.makedirs(output_dir, exist_ok=True)
    # job_idx = int(os.getenv("BEAKER_REPLICA_RANK"))
    # job_start_shard_idx = start_shard_idx + (job_idx * job_batch_size)
    # job_end_shard_idx = start_shard_idx + ((job_idx + 1) * job_batch_size)
    # print(f"{job_start_shard_idx=}")
    # print(f"{job_end_shard_idx=}")
    # data_shard_paths = sorted(glob.glob(source_dir + "/*.jsonl.gz"))[
    #     job_start_shard_idx : job_end_shard_idx
    #     + 1  # forgot to add 1 here! so last shard in output file is excluded
    # ]
    data_shard_paths = sorted(glob.glob(source_dir + "/*.jsonl.gz"))
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

    llm = LLM(
        model="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        task="embed",
        trust_remote_code=True,
        hf_overrides={"is_causal": True},
    )

    new_all_data = [
        process_jsonl(llm, data, batch_size, num_workers, level) for data in all_data
    ]

    for shard_path, data in zip(shard_paths, new_all_data):
        with gzip.open(f"{output_dir}/{os.path.basename(shard_path)}", "wt") as f:
            for seg in data:
                f.write(json.dumps(seg) + "\n")


if __name__ == "__main__":
    Fire(main)
    # main(source_dir="temp", output_dir="temp_2", batch_size=8192, num_workers=28)
