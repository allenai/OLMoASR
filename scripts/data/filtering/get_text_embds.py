from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
import os
from tqdm import tqdm
import torch
import json
from time import time
import gzip
import glob
import multiprocessing
from functools import partial
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Literal
from fire import Fire
import torch.nn.functional as F
import numpy as np
import webvtt
import sys

sys.path.append("/stage")
from open_whisper.utils import TranscriptReader
from whisper.normalizers import EnglishTextNormalizer


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
    return data


def init_tokenizer(worker_id: int, llm):
    global tokenizer
    tokenizer = llm.llm_engine.get_tokenizer_group()


def collate_fn(batch):
    video_id, encoded_man = batch # , encoded_mach = zip(*batch)
    encoded_man = [TokensPrompt(prompt_token_ids=item) for item in encoded_man]
    # encoded_mach = [TokensPrompt(prompt_token_ids=item) for item in encoded_mach]

    return video_id, encoded_man  # , encoded_mach


class SamplesDictsDataset(Dataset):
    def __init__(self, data, level):
        self.data = data
        self.level = level
        self.normalizer = EnglishTextNormalizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        man_text = (
            self.data[idx]["seg_text"]
            if self.level == "seg"
            else self.get_man_text(self.data[idx]["full_file"])
        )
        video_id = self.data[idx]["id"]
        # mach_text = (
        #     self.data[idx]["mach_seg_text"]
        #     if self.level == "seg"
        #     else self.get_mach_text(self.data[idx]["mach_content"])
        # )

        try:
            norm_man_text = self.normalizer(man_text)
        except Exception:
            norm_man_text = man_text

        # try:
        #     norm_mach_text = self.normalizer(mach_text)
        # except Exception:
        #     norm_mach_text = mach_text

        if self.level == "doc":
            encoded_man = tokenizer.encode(norm_man_text)
            # encoded_mach = tokenizer.encode(norm_mach_text)

            if len(encoded_man) >= 32768:
                encoded_man = encoded_man[:32768]

            # if len(encoded_mach) >= 32768:
            #     encoded_mach = encoded_mach[:32768]
            return video_id, encoded_man  # , encoded_mach
        else:
            if norm_man_text == "":
                norm_man_text = " "

            # if norm_mach_text == "":
            #     norm_mach_text = " "

            return video_id, norm_man_text  # , norm_mach_text

    def get_man_text(self, man_path):
        reader = TranscriptReader(
            file_path=man_path,
            transcript_string=None,
            ext="vtt" if man_path.endswith(".vtt") else "srt",
        )
        t_dict, *_ = reader.read()
        man_text = reader.extract_text(t_dict)
        return man_text.strip()

    # def get_mach_text(self, mach_content):
    #     mach_text = ""
    #     if mach_content != "":
    #         content = webvtt.from_string(mach_content)
    #         modified_content = []
    #         if len(content) > 0:
    #             if len(content) > 1:
    #                 if content[0].text == content[1].text:
    #                     modified_content.append(content[0])
    #                     start = 2
    #                 else:
    #                     start = 0
    #             elif len(content) == 1:
    #                 start = 0

    #             for i in range(start, len(content)):
    #                 caption = content[i]
    #                 if "\n" not in caption.text:
    #                     modified_content.append(caption)
    #                 elif "\n" in caption.text and i == len(content) - 1:
    #                     caption.text = caption.text.split("\n")[-1]
    #                     modified_content.append(caption)

    #             mach_text = " ".join([caption.text for caption in modified_content])

    #     return mach_text.strip()


def process_jsonl(llm, data, batch_size, num_workers, level, output_dir):
    if level == "doc":
        partial_init_tokenizer = partial(init_tokenizer, llm=llm)

    dataset = SamplesDictsDataset(data, level)

    if level == "doc":
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            worker_init_fn=partial_init_tokenizer,
            collate_fn=collate_fn,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )

    # scores = []
    embds = []
    video_ids = []
    start = time()
    for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        # man_input, mach_input = batch
        video_id, man_input = batch

        man_output = llm.embed(man_input)
        # mach_output = llm.embed(mach_input)

        man_embeds = F.normalize(
            torch.stack(
                [torch.tensor(output.outputs.embedding) for output in man_output], dim=0
            ),
            p=2,
            dim=-1,
        )

        embds.extend(man_embeds)
        video_ids.extend(video_id)

        if batch_idx % 2 == 0:
            result = dict(zip(video_ids, embds))
            torch.save(
                result, f"{output_dir}/text_embds_{batch_idx}.pt"
            )
            del result
            embds = []
            video_ids = []
        elif batch_idx == len(dataloader) - 1:
            result = dict(zip(video_ids, embds))
            torch.save(
                torch.stack(embds, dim=0), f"{output_dir}/text_embds_{batch_idx}.pt"
            )
            del result
            embds = []
            video_ids = []

        # mach_embeds = F.normalize(
        #     torch.stack(
        #         [torch.tensor(output.outputs.embedding) for output in mach_output],
        #         dim=0,
        #     ),
        #     p=2,
        #     dim=-1,
        # )

        # batch_score = F.cosine_similarity(man_embeds, mach_embeds, dim=-1)
        # print(f"{batch_score.shape=}")
        # batch_score = batch_score.tolist()

        # scores.extend(batch_score)

    print(f"Time taken: {time() - start}")
    # print(f"Number of scores: {len(scores)}")
    # print(f"Number of segments: {len(data)}")
    # assert len(scores) == len(data)

    # for i, sample in enumerate(data):
    #     if level == "doc":
    #         sample["sim_score"] = scores[i]
    #     else:
    #         sample["seg_sim_score"] = scores[i]

    # return data


def main(
    source_file: str,
    output_dir: str,
    batch_size: int,
    num_workers: int,
    level: str,
):
    os.makedirs(output_dir, exist_ok=True)
    data = open_file(source_file)
    print(f"{len(data)=}")

    llm = LLM(
        model="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        task="embed",
        trust_remote_code=True,
        hf_overrides={"is_causal": True},
    )

    process_jsonl(llm, data, batch_size, num_workers, level, output_dir)


if __name__ == "__main__":
    Fire(main)
    # main(
    #     source_dir="temp_doc",
    #     output_dir="temp_doc_2",
    #     batch_size=950,
    #     num_workers=28,
    #     level="doc",
    # )
