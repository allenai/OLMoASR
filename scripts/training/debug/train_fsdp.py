import os
import glob
import json
import numpy as np
import wandb
from typing import List, Tuple, Union, Optional, Dict, Literal
import time
import jiwer
from fire import Fire
from tqdm import tqdm
import multiprocessing
from itertools import chain
from collections import defaultdict
import functools
import subprocess
import gzip

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch.autograd import set_detect_anomaly
from torch.profiler import (
    profile,
    ProfilerActivity,
    record_function,
    schedule,
    tensorboard_trace_handler,
)

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast

import torch.multiprocessing as mp
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    FullStateDictConfig,
    FullOptimStateDictConfig,
    StateDictType,
    sharded_grad_scaler,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

import whisper
from whisper import audio, DecodingOptions
from whisper.normalizers import EnglishTextNormalizer
from whisper.tokenizer import get_tokenizer
import whisper.tokenizer
from open_whisper.config.model_dims import VARIANT_TO_DIMS, ModelDimensions
import open_whisper as ow
from whisper.model import (
    ResidualAttentionBlock,
    AudioEncoder,
    TextDecoder,
    MultiHeadAttention,
)

WANDB_EXAMPLES = 8
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
VARIANT_TO_PARAMS = {
    "tiny": 39 * 10**6,
    "base": 74 * 10**6,
    "small": 244 * 10**6,
    "medium": 769 * 10**6,
    "large": 1550 * 10**6,
}

HARDWARE_TO_FLOPS = {"H100": 900 * 10**12, "L40": 366 * 10**12, "A100": 312 * 10**12}


class AudioTextDataset(Dataset):
    """Dataset for audio and transcript segments

    Attributes:
        audio_files: List of audio file paths
        transcript_files: List of transcript file paths
        n_text_ctx: Number of text tokens
    """

    def __init__(
        self,
        samples: List[Dict],
        n_text_ctx: int,
        n_head: int,
    ):
        self.samples = samples
        self.n_text_ctx = n_text_ctx
        self.n_head = n_head

    def __len__(self):
        return len(self.samples)

    def __getitem__(
        self, index
    ) -> Tuple[str, str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # not sure if putting it here is bad...
        start_preproc = time.time()
        global tokenizer
        sample_dict = self.samples[index]
        audio_file = sample_dict["audio_file"].replace("ow_full", "ow_seg")
        transcript_file = sample_dict["subtitle_file"].replace("ow_full", "ow_seg")
        transcript_string = sample_dict["seg_content"]
        audio_input, padded_audio_arr = self.preprocess_audio(audio_file)
        text_input, text_y, padding_mask = self.preprocess_text(
            transcript_string, transcript_file, tokenizer
        )
        end_preproc = time.time()
        preproc_time = end_preproc - start_preproc

        return (
            audio_file,
            transcript_file,
            padded_audio_arr,
            audio_input,
            text_input,
            text_y,
            padding_mask,
            preproc_time,
        )

    def preprocess_audio(self, audio_file: str) -> Tuple[str, torch.Tensor]:
        """Preprocesses the audio data for the model.

        Loads the audio file, pads or trims the audio data, and computes the log mel spectrogram.

        Args:
            audio_file: The path to the audio file

        Returns:
            A tuple containing the name of audio file and the log mel spectrogram
        """
        audio_arr = np.load(audio_file).astype(np.float32) / 32768.0
        audio_arr = audio.pad_or_trim(audio_arr)
        mel_spec = audio.log_mel_spectrogram(audio_arr)

        return mel_spec, audio_arr

    def preprocess_text(
        self,
        transcript_string: str,
        transcript_file: str,
        tokenizer: whisper.tokenizer.Tokenizer,
    ) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocesses the text data for the model.

        Reads in the transcript file and extracts the text data. Tokenizes the text data and pads it to the context length.

        Args:
            transcript_file: The path to the transcript file
            tokenizer: The tokenizer to use for encoding the text data

        Returns:
            A tuple containing the transcript file, the input text tensor, the target text tensor, and the padding mask
        """
        # transcript -> text
        reader = ow.utils.TranscriptReader(
            transcript_string=transcript_string,
            file_path=None,
            ext=transcript_file.split(".")[-1],
        )
        transcript, *_ = reader.read()

        if not transcript:
            text_tokens = [tokenizer.no_speech]
        else:
            transcript_text = reader.extract_text(transcript=transcript)

            text_tokens = tokenizer.encode(transcript_text)

        text_tokens = (
            list(tokenizer.sot_sequence_including_notimestamps)
            + text_tokens
            + [tokenizer.eot]
        )

        # offset
        text_input = text_tokens[:-1]
        text_y = text_tokens[1:]

        padding_mask = torch.zeros((self.n_text_ctx, self.n_text_ctx))
        padding_mask[:, len(text_input) :] = -np.inf
        # causal_mask = (
        #     torch.empty(self.n_text_ctx, self.n_text_ctx).fill_(-np.inf).triu_(1)
        # )
        # padding_mask = padding_mask + causal_mask
        # padding_mask = padding_mask.unsqueeze(dim=0).repeat(self.n_head, 1, 1)[
        #     :, : self.n_text_ctx, : self.n_text_ctx
        # ]

        text_input = np.pad(
            text_input,
            pad_width=(0, self.n_text_ctx - len(text_input)),
            mode="constant",
            constant_values=51864,
        )
        text_y = np.pad(
            text_y,
            pad_width=(0, self.n_text_ctx - len(text_y)),
            mode="constant",
            constant_values=51864,
        )

        text_input = torch.tensor(text_input, dtype=torch.long)
        text_y = torch.tensor(text_y, dtype=torch.long)

        return text_input, text_y, padding_mask


def init_tokenizer(worker_id: int):
    global tokenizer
    tokenizer = get_tokenizer(multilingual=False)


def setup(rank: int) -> None:
    """Initializes the distributed process group

    Args:
        rank: The rank of the current process
        world_size: The total number of processes
    """
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")


def open_dicts_file(samples_dicts_file) -> List[Dict]:
    if samples_dicts_file.endswith(".gz"):
        with gzip.open(samples_dicts_file, "rt") as f:
            samples_dicts = [json.loads(line.strip()) for line in f]
    return samples_dicts


def prepare_dataloader(
    dataset: Dataset,
    batch_size: int,
    pin_memory: bool,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
) -> Tuple[DistributedSampler, DataLoader]:
    """Prepares the dataloader for the dataset

    Prepares the distributed sampler and the dataloader for the dataset for DDP training

    Args:
        dataset: The dataset to use
        rank: The rank of the current process
        world_size: The total number of processes
        batch_size: The batch size
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle the data
        num_workers: The number of workers
        persistent_workers: Whether to use persistent workers

    Returns:
        A tuple containing the distributed sampler and the dataloader
    """
    sampler = DistributedSampler(
        dataset,
        shuffle=shuffle,
        seed=42,
        drop_last=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        sampler=sampler,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=init_tokenizer,
    )

    return dataloader, sampler


def prepare_data(
    samples_dicts: List[Dict],
    train_batch_size: int,
    n_text_ctx: int,
    n_head: int,
    pin_memory: bool = True,
    shuffle: bool = True,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Prepares the data for training

    Given the list of audio and transcript files, prepares the distributed sampler and dataloader for training and validation
    If subset is not None, only uses a subset of the data

    Args:
        rank: The rank of the current process
        world_size: The total number of processes
        audio_files_train: The list of audio files for training
        transcript_files_train: The list of transcript files for training
        train_val_split: The ratio of training to validation data
        train_batch_size: The batch size for training
        val_batch_size: The batch size for validation
        n_text_ctx: The number of text tokens
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle the data
        num_workers: The number of workers
        persistent_workers: Whether to use persistent workers
        subset: The subset of the data to use

    Returns:
        A tuple containing the dataloaders for training and validation
    """
    audio_text_dataset = AudioTextDataset(
        samples=samples_dicts,
        n_text_ctx=n_text_ctx,
        n_head=n_head,
    )

    train_dataloader, train_sampler = prepare_dataloader(
        dataset=audio_text_dataset,
        batch_size=train_batch_size,
        pin_memory=pin_memory,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    return train_dataloader, train_sampler


def prepare_optim(
    model: torch.nn.Module,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Prepares the optimizer for training

    Prepares the AdamW optimizer for training

    Args:
        model: The model to train
        lr: The learning rate
        betas: The betas for the Adam optimizer
        eps: The epsilon value
        weight_decay: The weight decay

    Returns:
        The optimizer for training
    """
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    return optimizer


def prepare_sched(
    train_steps: int,
    world_size: int,
    train_batch_size: int,
    eff_batch_size: int,
    optimizer: torch.optim.Optimizer,
) -> Tuple[LambdaLR, int, int, int]:
    """Prepares the scheduler for training

    Prepares the LambdaLR scheduler for training

    Args:
        train_dataloader: The training dataloader
        world_size: The total number of processes
        train_batch_size: The batch size for training
        eff_batch_size: The effective train batch size
        optimizer: The optimizer for training

    Returns:
        A tuple containing the scheduler, the number of steps over which to accumulate gradients, the number of warmup steps, and the total number of steps
    """
    if eff_batch_size <= (world_size * train_batch_size):
        accumulation_steps = 1
    else:
        accumulation_steps = eff_batch_size // (
            world_size * train_batch_size
        )  # Number of steps over which to accumulate gradients
    warmup_steps = np.ceil(0.002 * train_steps)

    def lr_lambda(global_step: int) -> float:
        if global_step < warmup_steps:
            return float(global_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(train_steps - global_step)
            / float(max(1, train_steps - warmup_steps)),
        )

    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

    return scheduler, accumulation_steps, warmup_steps, train_steps


def setup_wandb(
    run_id: Optional[str],
    exp_name: str,
    job_type: str,
    model_variant: str,
    model_dims: ModelDimensions,
    train_steps: int,
    epoch_steps: int,
    warmup_steps: int,
    accumulation_steps: int,
    world_size: int,
    num_workers: int,
    prefetch_factor: int,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    eff_batch_size: int,
    train_batch_size: int,
    hardware: str,
    wandb_tags: List[str],
) -> Tuple[Optional[str], List[str], wandb.Artifact, wandb.Artifact, bool, bool]:
    """Sets up the Weights and Biases logging

    Args:
        run_id: The run ID
        exp_name: The experiment name
        job_type: The type of job
        subset: The subset of the data
        model_variant: The variant of the model
        lr: The learning rate
        betas: The betas for the Adam optimizer
        eps: The epsilon value
        weight_decay: The weight decay
        train_batch_size: The batch size for training
        total_steps: The total number of steps
        warmup_steps: The number of warmup steps
        accumulation_steps: The number of steps over which to accumulate gradients
        train_val_split: The ratio of training to validation data
        world_size: The total number of processes
        num_workers: The number of workers

    Returns:
        A tuple containing the run ID, the tags, the training results artifact, the validation results artifact,
        a boolean indicating whether the training results artifact has been added, and a boolean indicating whether the validation results artifact has been added
    """
    config = {
        "lr": lr,
        "betas": betas,
        "eps": eps,
        "weight_decay": weight_decay,
        "eff_batch_size": eff_batch_size,
        "train_batch_size": train_batch_size,
        "train_steps": train_steps,
        "epoch_steps": epoch_steps,
        "warmup_steps": warmup_steps,
        "accumulation_steps": accumulation_steps,
        "world_size": world_size,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "model_variant": model_variant,
        "n_mels": model_dims.n_mels,
        "n_audio_ctx": model_dims.n_audio_ctx,
        "n_audio_state": model_dims.n_audio_state,
        "n_audio_head": model_dims.n_audio_head,
        "n_audio_layer": model_dims.n_audio_layer,
        "n_vocab": model_dims.n_vocab,
        "n_text_ctx": model_dims.n_text_ctx,
        "n_text_state": model_dims.n_text_state,
        "n_text_head": model_dims.n_text_head,
        "n_text_layer": model_dims.n_text_layer,
        "model_params": VARIANT_TO_PARAMS[model_variant],
        "peak_flops": HARDWARE_TO_FLOPS[hardware],
    }

    if run_id is None:
        run_id = wandb.util.generate_id()

    wandb.init(
        id=run_id,
        resume="allow",
        project="open_whisper",
        entity="dogml",
        config=config,
        save_code=True,
        job_type=job_type,
        tags=(wandb_tags),
        name=exp_name,
        settings=wandb.Settings(init_timeout=300, _service_wait=300),
    )

    wandb.define_metric("global_step")
    wandb.define_metric("local_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("efficiency/time_per_step=*", step_metric="global_step")
    wandb.define_metric(
        "efficiency/audio_min_per_GPU_second_gpu=*", step_metric="global_step"
    )
    wandb.define_metric("efficiency/optim_step_time=*", step_metric="global_step")
    wandb.define_metric("efficiency/dl_time=*", step_metric="local_step")
    wandb.define_metric("efficiency/fwd_time=*", step_metric="local_step")
    wandb.define_metric("efficiency/bwd_time=*", step_metric="local_step")
    wandb.define_metric("efficiency/pass_time=*", step_metric="local_step")
    wandb.define_metric("efficiency/preproc_time=*", step_metric="local_step")

    return run_id


def train(
    rank: int,
    local_rank: int,
    global_step: int,
    local_step: int,
    train_batch_size: int,
    train_dataloader: DataLoader,
    train_sampler: DistributedSampler,
    train_steps: int,
    epoch_steps: int,
    epoch: int,
    scaler: GradScaler,
    model,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    accumulation_steps: int,
    max_grad_norm: float,
    verbose: bool,
    precision: torch.dtype,
    log_dir: str,
):
    total_loss = 0.0
    model.train()
    optimizer.zero_grad()

    train_sampler.set_epoch(epoch)
    start_dl = time.time()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    ) as prof:
        for batch_idx, batch in enumerate(train_dataloader):
            end_dl = time.time()
            model.train()

            if batch_idx % accumulation_steps == 0 or accumulation_steps == 1:
                start_step = time.time()

            with autocast(device_type="cuda", dtype=precision):
                (
                    *_,
                    audio_input,
                    text_input,
                    text_y,
                    padding_mask,
                    preproc_time,
                ) = batch

                start_data_to_gpu = time.time()
                audio_input = audio_input.to(local_rank)
                text_input = text_input.to(local_rank)
                text_y = text_y.to(local_rank)
                padding_mask = padding_mask.to(local_rank)
                end_data_to_gpu = time.time()

                start_fwd = time.time()
                logits = model(audio_input, text_input, padding_mask, verbose)
                end_fwd = time.time()

                train_loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    text_y.view(-1),
                    ignore_index=51864,
                )
                train_loss = (
                    train_loss / accumulation_steps
                )  # normalization of loss (gradient accumulation)

            start_bwd = time.time()
            scaler.scale(train_loss).backward()
            end_bwd = time.time()
            local_step += 1

            if rank == 0:
                wandb.log(
                    {
                        "efficiency/bwd_time": end_bwd - start_bwd,
                        "efficiency/dl_time": end_dl - start_dl,
                        "efficiency/data_to_gpu_time": end_data_to_gpu
                        - start_data_to_gpu,
                        "efficiency/fwd_time": end_fwd - start_fwd,
                        "efficiency/avg_preproc_time": sum(preproc_time)
                        / len(preproc_time),
                        "local_step": local_step,
                    }
                )

            train_loss.detach_()
            total_loss += train_loss

            # after accumulation_steps, update weights
            if ((batch_idx + 1) % accumulation_steps) == 0:
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                train_loss_all = total_loss.item() / dist.get_world_size()
                # train_loss_all = 0.0

                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)
                start_optim_step = time.time()
                scaler.step(optimizer)
                end_optim_step = time.time()
                if rank == 0:
                    wandb.log(
                        {
                            "efficiency/optim_step_time": end_optim_step
                            - start_optim_step,
                            "global_step": global_step,
                        }
                    )
                scaler.update()
                global_step += 1
                end_step = time.time()
                time_per_step = end_step - start_step
                throughput = (
                    (train_batch_size * accumulation_steps * 30) / 60
                ) / time_per_step

                if rank == 0:
                    wandb.log(
                        {
                            "efficiency/time_per_step": time_per_step,
                            "global_step": global_step,
                        }
                    )
                    wandb.log(
                        {
                            "efficiency/audio_min_per_GPU_second_gpu": throughput,
                            "global_step": global_step,
                        }
                    )

                current_lr = optimizer.param_groups[0]["lr"]
                # logging learning rate
                if rank == 0:
                    wandb.log(
                        {"train/learning_rate": current_lr, "global_step": global_step}
                    )
                scheduler.step()  # Adjust learning rate based on accumulated steps
                prof.step()

                if global_step >= epoch_steps + (epoch_steps * epoch):
                    # logging
                    if rank == 0:
                        train_metrics = defaultdict(float)
                        print("Logging results at an epoch")
                        print(f"global_step: {global_step}")
                        print(f"train_loss: {train_loss_all}")

                        train_metrics["train/train_loss"] = train_loss_all
                        train_metrics["global_step"] = global_step

                        wandb.log(train_metrics)

                    return (
                        global_step,
                        local_step,
                        epoch,
                        model,
                        optimizer,
                        scaler,
                        scheduler,
                    )

                if global_step >= train_steps:
                    # logging
                    if rank == 0:
                        train_metrics = defaultdict(float)
                        print("Logging final training results")
                        print(f"global_step: {global_step}")
                        print(f"train_loss: {train_loss_all}")

                        train_metrics["train/train_loss"] = train_loss_all
                        train_metrics["global_step"] = global_step

                        wandb.log(train_metrics)

                    return (
                        global_step,
                        local_step,
                        epoch,
                        model,
                        optimizer,
                        scaler,
                        scheduler,
                    )

                optimizer.zero_grad()  # Reset gradients only after updating weights
                total_loss = 0.0

                if rank == 0:
                    train_metrics = defaultdict(float)
                    print(f"global_step: {global_step}")
                    print(f"train_loss: {train_loss_all}")

                    train_metrics["train/train_loss"] = train_loss_all
                    train_metrics["global_step"] = global_step

                    wandb.log(train_metrics)

            end_pass = time.time()
            start_dl = time.time()

            if rank == 0:
                wandb.log(
                    {
                        "efficiency/pass_time": end_pass - start_fwd,
                        "local_step": batch_idx + 1,
                    }
                )


def cleanup():
    """Cleanup function for the distributed training"""
    torch.cuda.empty_cache()
    dist.destroy_process_group()


def main(
    model_variant: str,
    exp_name: str,
    job_type: str,
    samples_dicts_dir: str,
    train_steps: int,
    epoch_steps: int,
    lr: float = 1.5e-3,
    betas: tuple = (0.9, 0.98),
    eps: float = 1e-6,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    eff_batch_size: int = 256,
    train_batch_size: int = 8,
    num_workers: int = 10,
    pin_memory: bool = True,
    shuffle: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    verbose: bool = False,
    precision: Literal["bfloat16", "float16", "pure_float16", "float32"] = "bfloat16",
    sharding_strategy: Literal[
        "FULL_SHARD",
        "SHARD_GRAD_OP",
        "HYBRID_SHARD",
        "_HYBRID_SHARD_ZERO2",
    ] = "FULL_SHARD",
    hardware: str = "H100",
    log_dir: str = "logs",
):
    run_id = None
    tags = ["debug"]
    model_dims = VARIANT_TO_DIMS[model_variant]
    os.makedirs(log_dir, exist_ok=True)

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    # setup the process groups
    setup(local_rank)

    if rank == 0:
        print(
            f"""
            {model_variant=},
            {train_steps=},
            {epoch_steps=},
            {local_rank=}, 
            {local_world_size=}, 
            {rank=}, 
            {world_size=}, 
            {dist.get_rank()=}, 
            {dist.get_world_size()=}, 
            {int(os.getenv('GROUP_RANK'))}
            """
        )

    # setup the tokenizer and normalizer
    tokenizer = get_tokenizer(multilingual=False)
    normalizer = EnglishTextNormalizer()
    n_text_ctx = model_dims.n_text_ctx
    n_head = model_dims.n_text_head

    # load samples dicts
    samples_dicts_files = glob.glob(f"{samples_dicts_dir}/*.jsonl.gz")

    # loading in data paths
    with multiprocessing.Pool() as pool:
        samples_dicts = list(
            chain(
                *tqdm(
                    pool.imap_unordered(open_dicts_file, samples_dicts_files),
                    total=len(samples_dicts_files),
                )
            )
        )
    if rank == 0:
        print(f"{len(samples_dicts)=}")

    # data preparation
    train_dataloader, train_sampler = prepare_data(
        samples_dicts=samples_dicts,
        train_batch_size=train_batch_size,
        n_text_ctx=n_text_ctx,
        n_head=n_head,
        pin_memory=pin_memory,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    print(f"Rank: {rank}, {len(train_dataloader)=}")

    # model precision
    if precision == "float16":
        precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float16,
        )
        autocast_precision = torch.float16
    elif precision == "float32":
        precision_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        autocast_precision = torch.float32
    elif precision == "pure_float16":
        precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
        autocast_precision = torch.float16
    elif precision == "bfloat16":
        precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        autocast_precision = torch.bfloat16
    
    if rank == 0:
        print(f"{precision_policy=}")
        print(f"{autocast_precision=}")

    # sharding strategy
    if sharding_strategy == "FULL_SHARD":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "SHARD_GRAD_OP":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif sharding_strategy == "HYBRID_SHARD":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "_HYBRID_SHARD_ZERO2":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    if rank == 0:
        print(f"{sharding_strategy=}")
    
    # model instantiation
    model = ow.model.Whisper(dims=model_dims).to(local_rank)

    size_model = 0
    if rank == 0:
        for i, (name, param) in enumerate(model.named_parameters()):
            if param.data.is_floating_point():
                size_model += param.numel() * torch.finfo(param.data.dtype).bits
            else:
                size_model += param.numel() * torch.iinfo(param.data.dtype).bits
        print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            ResidualAttentionBlock,
            AudioEncoder,
            TextDecoder,
            MultiHeadAttention,
        },
    )
    model = FSDP(
        model,
        device_id=local_rank,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=precision_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=sharding_strategy,
    )

    # optimizer and scheduler instantiation
    optimizer = prepare_optim(
        model=model,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    scheduler, accumulation_steps, warmup_steps, train_steps = prepare_sched(
        train_steps=train_steps,
        world_size=world_size,
        train_batch_size=train_batch_size,
        eff_batch_size=eff_batch_size,
        optimizer=optimizer,
    )

    # https://github.com/pytorch/pytorch/issues/76607
    # if using FSDP mixed precision w/ fp16, need to use sharded grad scaler
    if precision == "float16" or precision == "pure_float16":
        scaler = sharded_grad_scaler.ShardedGradScaler()
    elif precision == "float32":
        scaler = None
    else:
        scaler = GradScaler()

    global_step = 0
    local_step = 0
    epoch = 0

    # setting up activation checkpointing
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, ResidualAttentionBlock)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )

    # setting up wandb for logging
    if rank == 0:
        run_id = setup_wandb(
            run_id=run_id,
            exp_name=exp_name,
            job_type=job_type,
            model_variant=model_variant,
            model_dims=model_dims,
            train_steps=train_steps,
            epoch_steps=epoch_steps,
            warmup_steps=warmup_steps,
            accumulation_steps=accumulation_steps,
            world_size=world_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            eff_batch_size=eff_batch_size,
            train_batch_size=train_batch_size,
            hardware=hardware,
            wandb_tags=tags,
        )

    while global_step < train_steps:
        global_step, local_step, epoch, model, optimizer, scaler, scheduler = train(
            rank=rank,
            local_rank=local_rank,
            global_step=global_step,
            local_step=local_step,
            train_batch_size=train_batch_size,
            train_dataloader=train_dataloader,
            train_sampler=train_sampler,
            train_steps=train_steps,
            epoch_steps=epoch_steps,
            epoch=epoch,
            scaler=scaler,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            accumulation_steps=accumulation_steps,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            precision=autocast_precision,
            log_dir=log_dir,
        )

        epoch += 1

    cleanup()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    Fire(main)
