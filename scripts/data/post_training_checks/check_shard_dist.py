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


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train(rank, world_size, shards):
    setup(rank, world_size)

    dataset = wds.DataPipeline(
        wds.SimpleShardList(shards),
        wds.split_by_worker,
        wds.shuffle(bufsize=1000, initial=100),
    )

    for epoch in range(2):
        print(f"{epoch=}")
        for shard in dataset:
            print(f"{rank=} {shard=}")


def main(shards: str):
    world_size = torch.cuda.device_count()
    print(world_size)
    mp.spawn(train, args=(world_size, shards), nprocs=world_size, join=True)


if __name__ == "__main__":
    Fire(main)
