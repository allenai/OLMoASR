import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import webdataset as wds
from torch.utils.data import DataLoader
from fire import Fire
import json
from typing import Dict, Tuple
import numpy as np


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def decode_sample(sample: Dict[str, bytes]):
    """
    Decodes a sample dictionary containing audio and text data.

    Args:
        sample: A dictionary containing audio and text data.
            The dictionary should have the following keys:
            - "__url__": The URL of the file.
            - "__key__": The key of the file.
            - "npy": The audio data in bytes.
            - "srt": The text data in bytes.

    Returns:
        A tuple containing the following elements:
            - audio_path (str): The path to the audio file.
            - audio_arr (np.ndarray): The audio data as a NumPy array.
            - text_path (str): The path to the text file.
            - transcript_str (str): The decoded text transcript.
    """
    # file_path = os.path.join(sample["__url__"], sample["__key__"])
    # audio_path = file_path + ".npy"
    # text_path = file_path + ".srt"
    try:
        audio_bytes = sample["npy"]
        text_bytes = sample["srt"]
        # audio_arr = decode_audio_bytes(audio_bytes)
        # transcript_str = decode_text_bytes(text_bytes)
    except KeyError:
        with open("error_wds.txt", "a") as f:
            f.write(sample["__key__"] + "\t" + sample["__url__"] + "\n")
        
    return list(sample.keys())
    
def train(rank, world_size, shards, batch_size):
    setup(rank, world_size)

    dataset = wds.DataPipeline(
        wds.SimpleShardList(shards),
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.map(decode_sample),
        wds.batched(batch_size)
    )

    idx = 0
    for sample in dataset:
        idx += 1
        print(f"{idx=}")
        print(f"{rank=} {sample=}")


def main(shards: str, batch_size: int):
    world_size = torch.cuda.device_count()
    print(world_size)
    mp.spawn(train, args=(world_size, shards, batch_size), nprocs=world_size, join=True)


if __name__ == "__main__":
    Fire(main)
