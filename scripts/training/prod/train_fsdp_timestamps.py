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
import zstandard as zstd
import io

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

from scripts.eval.eval import EvalDataset
from for_logging import TRAIN_TABLE_COLS, EVAL_TABLE_COLS, VAL_TABLE_COLS

from datasets import load_dataset
import librosa

WANDB_EXAMPLES = 8
os.environ["WANDB__SERVICE_WAIT"] = "300"
VARIANT_TO_PARAMS = {
    "tiny": 39 * 10**6,
    "base": 74 * 10**6,
    "small": 244 * 10**6,
    "medium": 769 * 10**6,
    "large": 1550 * 10**6,
}

HARDWARE_TO_FLOPS = {
    "H100": 900 * 10**12,
    "L40": 366 * 10**12,
    "A100": 312 * 10**12,
    "A6000": 310 * 10**12,
}


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
        audio_file = sample_dict["audio_file"].replace("ow_seg", "ow_seg_long")
        transcript_file = sample_dict["subtitle_file"].replace("ow_seg", "ow_seg_long")
        transcript_string = sample_dict["seg_content"]
        ts_mode = sample_dict["ts_mode"]
        only_no_ts_mode = sample_dict["only_no_ts_mode"]
        norm_end = sample_dict["norm_end"]
        # new_norm_end is temp fix for text segs w/ > 30s -> current don't know why issue occurs
        (
            text_input,
            text_y,
            padding_mask,
            timestamp_mode,
            new_norm_end,
            text_preproc_time,
        ) = self.preprocess_text(
            transcript_string,
            transcript_file,
            tokenizer,
            norm_end,
            ts_mode,
            only_no_ts_mode,
        )

        if timestamp_mode is True:
            norm_end = None
        elif timestamp_mode is False and (new_norm_end != norm_end):
            norm_end = new_norm_end

        audio_input, padded_audio_arr, audio_preproc_time, audio_load_time = (
            self.preprocess_audio(audio_file, norm_end)
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
            audio_preproc_time,
            audio_load_time,
            text_preproc_time,
        )

    def preprocess_audio(
        self, audio_file: str, norm_end: Optional[Union[str, int]]
    ) -> Tuple[str, torch.Tensor]:
        """Preprocesses the audio data for the model.

        Loads the audio file, pads or trims the audio data, and computes the log mel spectrogram.

        Args:
            audio_file: The path to the audio file

        Returns:
            A tuple containing the name of audio file and the log mel spectrogram
        """
        start_time = time.time()
        audio_arr = np.load(audio_file).astype(np.float32) / 32768.0
        audio_load_time = time.time() - start_time
        if norm_end:
            # number of samples to trim until
            if isinstance(norm_end, str):
                norm_end = ow.utils.convert_to_milliseconds(norm_end)

            length = norm_end * 16
            # trim until end of text segment
            audio_arr = audio.pad_or_trim(audio_arr, length=length)
            # pad w/ silence
            audio_arr = audio.pad_or_trim(audio_arr)
        else:
            # in case audio_arr isn't exactly 480K samples
            audio_arr = audio.pad_or_trim(audio_arr)
        mel_spec = audio.log_mel_spectrogram(audio_arr)
        audio_preproc_time = time.time() - start_time

        return mel_spec, audio_arr, audio_preproc_time, audio_load_time

    def preprocess_text(
        self,
        transcript_string: str,
        transcript_file: str,
        tokenizer: whisper.tokenizer.Tokenizer,
        norm_end: Union[int, str],
        ts_mode: bool,
        only_no_ts_mode: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Preprocesses the text data for the model.

        Reads in the transcript file and extracts the text data. Tokenizes the text data and pads it to the context length.

        Args:
            transcript_file: The path to the transcript file
            tokenizer: The tokenizer to use for encoding the text data

        Returns:
            A tuple containing the transcript file, the input text tensor, the target text tensor, and the padding mask
        """
        start_time = time.time()
        reader = ow.utils.TranscriptReader(
            transcript_string=transcript_string,
            file_path=None,
            ext=transcript_file.split(".")[-1],
        )
        transcript, *_ = reader.read()
        timestamp_mode = False

        if isinstance(norm_end, str):
            norm_end = ow.utils.convert_to_milliseconds(norm_end)

        if not transcript:
            if (
                norm_end > 30000
            ):  # can't rmb if this is a valid case but leaving in for now
                next_start_token_idx = [tokenizer.timestamp_begin + (30000 // 20)]
            else:
                next_start_token_idx = [tokenizer.timestamp_begin + (norm_end // 20)]

            if norm_end >= 30000:
                tokens = (
                    list(tokenizer.sot_sequence_including_notimestamps)
                    + [tokenizer.no_speech]
                    + [tokenizer.eot]
                )
            else:
                if only_no_ts_mode is True:
                    tokens = (
                        list(tokenizer.sot_sequence_including_notimestamps)
                        + tokenizer.encode("")
                        + [tokenizer.eot]
                    )
                else:
                    if np.random.rand() >= 0.5:
                        tokens = (
                            [tokenizer.sot_sequence[0]]
                            + [tokenizer.timestamp_begin]
                            + tokenizer.encode("")
                            + next_start_token_idx
                            + next_start_token_idx
                            + [tokenizer.eot]
                        )
                        timestamp_mode = True
                    else:
                        tokens = (
                            list(tokenizer.sot_sequence_including_notimestamps)
                            + tokenizer.encode("")
                            + [tokenizer.eot]
                        )
        else:
            if norm_end > 30000:  # temp soln
                if len(transcript) > 1:
                    del transcript[list(transcript.keys())[-1]]
                    norm_end = list(transcript.keys())[-1][1]
                only_no_ts_mode = True

            tokens = [
                (tokenizer.encode(" " + text.strip()))
                for i, (_, text) in enumerate(transcript.items())
            ]

            if only_no_ts_mode is True:
                tokens = (
                    list(tokenizer.sot_sequence_including_notimestamps)
                    + list(chain(*tokens))
                    + [tokenizer.eot]
                )
            else:
                if np.random.rand() >= 0.5:
                    if ts_mode is True:

                        def convert_to_token_idx(timestamp, timestamp_begin):
                            ts_ms = ow.utils.convert_to_milliseconds(timestamp)
                            if ts_ms > 30000:
                                return None
                            else:
                                return timestamp_begin + (ts_ms // 20)

                        timestamp_begin = tokenizer.timestamp_begin
                        sot_token = tokenizer.sot_sequence[0]

                        # Precompute start and end token indices
                        token_ranges = []
                        invalid = False

                        for start, end in transcript.keys():
                            start_idx = convert_to_token_idx(start, timestamp_begin)
                            end_idx = convert_to_token_idx(end, timestamp_begin)

                            # handling invalid timestamps (> 30s)
                            if start_idx is None or end_idx is None:
                                invalid = True
                                break
                            token_ranges.append((start_idx, end_idx))

                        if invalid is True:
                            # If any token range is None, skip this segment
                            tokens = (
                                list(tokenizer.sot_sequence_including_notimestamps)
                                + list(chain(*tokens))
                                + [tokenizer.eot]
                            )
                        else:
                            # Build new_tokens using list comprehension
                            new_tokens = [
                                (
                                    [sot_token] + [start] + tokens[i] + [end]
                                    if i == 0
                                    else [start] + tokens[i] + [end]
                                )
                                for i, (start, end) in enumerate(token_ranges)
                            ]

                            new_tokens = list(chain(*new_tokens))

                            if (
                                norm_end > 30000
                            ):  # can't rmb if this is a valid case but leaving in for now
                                next_start_token_idx = [
                                    tokenizer.timestamp_begin + (30000 // 20)
                                ]
                            else:
                                next_start_token_idx = [
                                    tokenizer.timestamp_begin + (norm_end // 20)
                                ]

                            new_tokens.extend(next_start_token_idx + [tokenizer.eot])
                            tokens = new_tokens
                            timestamp_mode = True
                    else:
                        tokens = (
                            list(tokenizer.sot_sequence_including_notimestamps)
                            + list(chain(*tokens))
                            + [tokenizer.eot]
                        )
                else:
                    tokens = (
                        list(tokenizer.sot_sequence_including_notimestamps)
                        + list(chain(*tokens))
                        + [tokenizer.eot]
                    )

        # offset
        text_input = tokens[:-1]
        text_y = tokens[1:]

        if len(text_input) > self.n_text_ctx:
            print(f"{transcript_file=}")
            print(f"{timestamp_mode=}")
            print(f"{norm_end=}")
            print(f"{transcript_string=}")
            print(f"{len(text_input)=}")
            print(f"{text_input=}")

        if len(text_y) > self.n_text_ctx:
            print(f"{transcript_file=}")
            print(f"{timestamp_mode=}")
            print(f"{norm_end=}")
            print(f"{transcript_string=}")
            print(f"{len(text_y)=}")
            print(f"{text_y=}")

        if max(tokens) >= 51864:
            print(f"{transcript_file=}")
            print(f"{timestamp_mode=}")
            print(f"{norm_end=}")
            print(f"{transcript_string=}")
            print("Invalid token index found:", max(tokens), "vs max allowed: 51863")

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
        text_preproc_time = time.time() - start_time

        return (
            text_input,
            text_y,
            padding_mask,
            timestamp_mode,
            norm_end,
            text_preproc_time,
        )


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
    elif samples_dicts_file.endswith(".zst"):
        samples_dicts = []
        with open(samples_dicts_file, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_stream:
                    try:
                        samples_dicts.append(json.loads(line))
                    except json.JSONDecodeError:
                        break  # reached padding at the end
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
    wandb.define_metric("val/*", step_metric="global_step")
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


def save_ckpt(
    rank: int,
    global_step: int,
    local_step: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    scheduler: LambdaLR,
    model_dims: ModelDimensions,
    tags: List,
    model_variant: str,
    exp_name: str,
    run_id: str,
    file_name: str,
    ckpt_dir: str,
) -> None:
    """Save model (DDP) checkpoint

    Saves non-DDP and DDP model checkpoints to checkpoints/{exp_name}_{run_id} directory in the format of {file_name}_{model_variant}_{tags}_{ddp}.pt

    Args:
        current_step: The current step
        best_val_loss: The best validation loss
        model: The model to save
        optimizer: The optimizer to save
        scaler: The gradient scaler to save
        scheduler: The scheduler to save
        model_dims: The model dimensions
        tags: The tags to use for logging
        model_variant: The variant of the model
        exp_name: The experiment name
        run_id: The run ID
        file_name: The file name
        ckpt_dir: Directory where all results are logged
    """
    # Prepare the FSDP checkpoint
    train_state = {
        "global_step": global_step,
        "local_step": local_step,
        "epoch": epoch,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "scheduler_state_dict": (scheduler.state_dict() if scheduler else None),
        "dims": model_dims,
    }

    state_dict_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    optim_state_dict_cfg = FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )

    # Save the full FSDP state dict
    with FSDP.state_dict_type(
        model,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=state_dict_cfg,
        optim_state_dict_config=optim_state_dict_cfg,
    ):
        model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model=model, optim=optimizer)

    # ckpt for eval
    eval_ckpt = {"model_state_dict": model_state, "dims": model_dims}

    if rank == 0:
        os.makedirs(f"{ckpt_dir}/{exp_name}_{run_id}", exist_ok=True)

    if file_name != "latesttrain" and rank == 0:
        if len(glob.glob(f"{ckpt_dir}/{exp_name}_{run_id}/*_{file_name}_*.pt")) > 0:
            for p in glob.glob(f"{ckpt_dir}/{exp_name}_{run_id}/*_{file_name}_*.pt"):
                if "eval" not in p:
                    os.remove(p)

    if rank == 0:
        torch.save(
            model_state,
            f"{ckpt_dir}/{exp_name}_{run_id}/model_state_{file_name}_{global_step:08}_{model_variant}_{'_'.join(tags)}.pt",
        )
        torch.save(
            optim_state,
            f"{ckpt_dir}/{exp_name}_{run_id}/optim_state_{file_name}_{global_step:08}_{model_variant}_{'_'.join(tags)}.pt",
        )
        torch.save(
            train_state,
            f"{ckpt_dir}/{exp_name}_{run_id}/train_state_{file_name}_{global_step:08}_{model_variant}_{'_'.join(tags)}.pt",
        )
        torch.save(
            eval_ckpt,
            f"{ckpt_dir}/{exp_name}_{run_id}/eval_{file_name}_{global_step:08}_{model_variant}_{'_'.join(tags)}.pt",
        )

    return f"{ckpt_dir}/{exp_name}_{run_id}/eval_{file_name}_{global_step:08}_{model_variant}_{'_'.join(tags)}.pt"


def load_ckpt(
    exp_name: str,
    run_id: str,
    rank: int,
    world_size: int,
    train_steps: int,
    train_batch_size: int,
    eff_batch_size: int,
    file_name: Optional[str],
    ckpt_dir: str,
    model_variant: str,
    precision: Literal["bfloat16", "float16", "pure_float16", "float32"],
    precision_policy: MixedPrecision,
    sharding_strategy: ShardingStrategy,
) -> Tuple[
    int,
    float,
    torch.nn.Module,
    torch.optim.Optimizer,
    GradScaler,
    LambdaLR,
    int,
    int,
    int,
]:
    """Loads the model (DDP) checkpoint to resume training

    Args:
        exp_name: The experiment name.
        run_id: The run ID
        rank: The rank of the current process
        world_size: The world size
        train_batch_size: The batch size for training.
        eff_size: The effective size
        train_dataloader: The training dataloader

    Returns:
        A tuple containing the current step, the best validation loss, the model, the optimizer, the gradient scaler,
        the scheduler, the number of steps over which to accumulate gradients, the number of warmup steps, and the total number of steps
    """
    map_location = "cpu"

    if file_name is "":
        all_train_state_files = glob.glob(
            f"{ckpt_dir}/{exp_name}_{run_id}/train_state_*_{model_variant}_*.pt"
        )
        latest_step = max(
            [int(f.split("/")[-1].split("_")[3]) for f in all_train_state_files]
        )
        train_state_file = glob.glob(
            f"{ckpt_dir}/{exp_name}_{run_id}/train_state_*_{latest_step:08}_{model_variant}_*.pt"
        )[0]
        model_state_file = glob.glob(
            f"{ckpt_dir}/{exp_name}_{run_id}/model_state_*_{latest_step:08}_{model_variant}_*.pt"
        )[0]
        optim_state_file = glob.glob(
            f"{ckpt_dir}/{exp_name}_{run_id}/optim_state_*_{latest_step:08}_{model_variant}_*.pt"
        )[0]
        print(f"{train_state_file=}")
        print(f"{model_state_file=}")
        print(f"{optim_state_file=}")
        # latest_ckpt_file = max(all_ckpt_files, key=os.path.getctime)

    train_state = torch.load(
        train_state_file, map_location=map_location, weights_only=False
    )

    # if end at training step i, then start at step i+1 when resuming
    global_step = train_state["global_step"]
    local_step = train_state["local_step"]

    epoch = train_state["epoch"]

    if precision == "float16" or precision == "pure_float16":
        scaler = sharded_grad_scaler.ShardedGradScaler()
        scaler.load_state_dict(train_state["scaler_state_dict"])
    else:
        # scaler = GradScaler(init_scale=2**16)
        scaler = None

    model = ow.model.Whisper(dims=train_state["dims"]).to(rank)
    model_state = torch.load(
        model_state_file, map_location=map_location, weights_only=False
    )
    model.load_state_dict(model_state)

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={ResidualAttentionBlock},
    )
    model = FSDP(
        model,
        device_id=rank,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=precision_policy,
        sync_module_states=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=sharding_strategy,
    )

    optimizer = AdamW(model.parameters())
    optim_state = torch.load(
        optim_state_file, map_location=map_location, weights_only=False
    )
    fsdp_optim_state = FSDP.optim_state_dict_to_load(
        model=model,
        optim=optimizer,
        optim_state_dict=optim_state,
    )
    optimizer.load_state_dict(fsdp_optim_state)

    scheduler, accumulation_steps, warmup_steps, train_steps = prepare_sched(
        train_steps=train_steps,
        world_size=world_size,
        train_batch_size=train_batch_size,
        eff_batch_size=eff_batch_size,
        optimizer=optimizer,
    )
    scheduler.load_state_dict(train_state["scheduler_state_dict"])

    return (
        global_step,
        local_step,
        epoch,
        model,
        optimizer,
        scaler,
        scheduler,
        accumulation_steps,
        warmup_steps,
        train_steps,
    )


def gen_pred(logits, text_y, tokenizer):
    probs = F.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)

    # collecting data for logging
    microbatch_pred_text = []
    microbatch_unnorm_pred_text = []
    for pred_instance in pred.cpu().numpy():
        pred_instance_text = tokenizer.decode_with_timestamps(list(pred_instance))
        microbatch_unnorm_pred_text.append(pred_instance_text)
        pred_instance_text = ow.utils.remove_after_endoftext(pred_instance_text)
        microbatch_pred_text.append(pred_instance_text)

    microbatch_tgt_text = []
    for text_y_instance in text_y.cpu().numpy():
        text_y_instance = list(filter(lambda token: token != 51864, text_y_instance))
        tgt_y_instance_text = tokenizer.decode_with_timestamps(list(text_y_instance))
        tgt_y_instance_text = tgt_y_instance_text.split("<|endoftext|>")[0]
        tgt_y_instance_text = tgt_y_instance_text + "<|endoftext|>"
        microbatch_tgt_text.append(tgt_y_instance_text)

    return microbatch_pred_text, microbatch_unnorm_pred_text, microbatch_tgt_text


def calc_pred_wer(batch_tgt_text, batch_pred_text, normalizer):
    norm_batch_tgt_text = [normalizer(text) for text in batch_tgt_text]
    norm_batch_pred_text = [normalizer(text) for text in batch_pred_text]
    norm_tgt_pred_pairs = list(zip(norm_batch_tgt_text, norm_batch_pred_text))
    # no empty references - for WER calculation
    batch_tgt_text_full = [
        norm_batch_tgt_text[i]
        for i in range(len(norm_batch_tgt_text))
        if len(norm_batch_tgt_text[i]) > 0
    ]
    batch_pred_text_full = [
        norm_batch_pred_text[i]
        for i in range(len(norm_batch_pred_text))
        if len(norm_batch_tgt_text[i]) > 0
    ]

    if len(batch_tgt_text_full) == 0 and len(batch_pred_text_full) == 0:
        train_wer = 0.0
        subs = 0
        dels = 0
        ins = 0
    else:
        train_wer = (
            jiwer.wer(
                reference=batch_tgt_text_full,
                hypothesis=batch_pred_text_full,
            )
            * 100
        )
        measures = jiwer.compute_measures(
            truth=batch_tgt_text_full, hypothesis=batch_pred_text_full
        )
        subs = measures["substitutions"]
        dels = measures["deletions"]
        ins = measures["insertions"]

    return (
        norm_tgt_pred_pairs,
        train_wer,
        subs,
        dels,
        ins,
    )


def log_tbl(
    global_step,
    train_table,
    run_id,
    batch_audio_files,
    batch_audio_arr,
    batch_text_files,
    batch_pred_text,
    batch_tgt_text,
    batch_unnorm_pred_text,
    norm_tgt_pred_pairs,
):
    for i, (
        tgt_text_instance,
        pred_text_instance,
    ) in enumerate(norm_tgt_pred_pairs):
        wer = np.round(
            ow.utils.calculate_wer((tgt_text_instance, pred_text_instance)),
            2,
        )
        subs = 0
        dels = 0
        ins = 0
        if len(tgt_text_instance) == 0:
            subs = 0
            dels = 0
            ins = len(pred_text_instance.split())
        else:
            measures = jiwer.compute_measures(tgt_text_instance, pred_text_instance)
            subs = measures["substitutions"]
            dels = measures["deletions"]
            ins = measures["insertions"]

        train_table.add_data(
            run_id,
            batch_audio_files[i],
            wandb.Audio(
                batch_audio_arr[i],
                sample_rate=16000,
            ),
            batch_text_files[i],
            pred_text_instance,
            batch_unnorm_pred_text[i],
            batch_pred_text[i],
            tgt_text_instance,
            batch_tgt_text[i],
            subs,
            dels,
            ins,
            len(tgt_text_instance.split()),
            wer,
        )

    wandb.log({f"train_table_{global_step}": train_table})


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
    scaler: Optional[GradScaler],
    model: FSDP,
    tokenizer: whisper.tokenizer.Tokenizer,
    normalizer: EnglishTextNormalizer,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    accumulation_steps: int,
    max_grad_norm: float,
    model_dims: Optional[ModelDimensions],
    model_variant: Optional[str],
    run_val: bool,
    val_freq: int,
    val_batch_size: int,
    val_num_workers: int,
    val_cache_dir: str,
    val_wandb_log: bool,
    run_eval: bool,
    run_id: Optional[str],
    tags: Optional[List[str]],
    exp_name: Optional[str],
    log_dir: str,
    ckpt_dir: str,
    train_log_freq: int,
    eval_freq: int,
    ckpt_freq: int,
    verbose: bool,
    precision: torch.dtype,
    eval_script_path: str,
    eval_dir: str,
    eval_wandb_log: bool,
    eval_batch_size: int,
    run_id_dir: Optional[str],
    eval_on_gpu: bool,
) -> Tuple[
    int,
    float,
    torch.nn.Module,
    torch.optim.Optimizer,
    Optional[GradScaler],
    LambdaLR,
    Optional[bool],
    Optional[bool],
]:
    """Training loop for 1 epoch

    Args:
        rank: The rank of the current process
        train_batch_size: The batch size for training
        train_dataloader: The dataloader for training
        scaler: The gradient scaler
        model: The model to train
        tokenizer: The tokenizer for encoding the text data
        normalizer: The text normalizer
        optimizer: The optimizer for training
        scheduler: The scheduler for training
        accumulation_steps: The number of steps over which to accumulate gradients
        max_grad_norm: The maximum gradient norm
        tags: The tags to use for logging
        exp_name: The experiment name

    Returns:
        A tuple containing the model, the optimizer, the gradient scaler, the scheduler, and a boolean indicating whether the training results artifact has been added
    """
    batch_pred_text = []
    batch_tgt_text = []
    batch_unnorm_pred_text = []
    batch_audio_files = []
    batch_text_files = []
    batch_audio_arr = []

    total_loss = 0.0
    model.train()
    optimizer.zero_grad()

    if rank == 0:
        train_table = wandb.Table(columns=TRAIN_TABLE_COLS)

    train_sampler.set_epoch(epoch)
    start_dl = time.time()

    for batch_idx, batch in enumerate(train_dataloader):
        end_dl = time.time()
        model.train()

        if batch_idx % accumulation_steps == 0 or accumulation_steps == 1:
            start_step = time.time()

        with autocast(device_type="cuda", dtype=precision):
            (
                audio_files,
                transcript_files,
                padded_audio_arr,
                audio_input,
                text_input,
                text_y,
                padding_mask,
                preproc_time,
                audio_preproc_time,
                audio_load_time,
                text_preproc_time,
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
        if scaler is not None:
            scaler.scale(train_loss).backward()  # might not need scaler
        else:
            train_loss.backward()
        end_bwd = time.time()
        local_step += 1

        if rank == 0:
            wandb.log(
                {
                    "efficiency/bwd_time": end_bwd - start_bwd,
                    "efficiency/dl_time": end_dl - start_dl,
                    "efficiency/data_to_gpu_time": end_data_to_gpu - start_data_to_gpu,
                    "efficiency/fwd_time": end_fwd - start_fwd,
                    "efficiency/avg_preproc_time": sum(preproc_time)
                    / len(preproc_time),
                    "efficiency/avg_audio_preproc_time": sum(audio_preproc_time)
                    / len(audio_preproc_time),
                    "efficiency/avg_audio_load_time": sum(audio_load_time)
                    / len(audio_load_time),
                    "efficiency/avg_text_preproc_time": sum(text_preproc_time)
                    / len(text_preproc_time),
                    "local_step": local_step,
                }
            )

        train_loss.detach_()
        total_loss += train_loss

        # alerting if loss is nan
        if rank == 0:
            if torch.isnan(train_loss):
                text = f"Loss is NaN for {audio_files} at step {global_step}!"
                print(f"{audio_input=}\n")
                print(f"{text_input=}\n")
                print(f"{text_y=}\n")
                print(f"{padding_mask=}\n")
                wandb.alert(title="NaN Loss", text=text)

        if ((global_step + 1) % train_log_freq) == 0:
            microbatch_pred_text, microbatch_unnorm_pred_text, microbatch_tgt_text = (
                gen_pred(
                    logits,
                    text_y,
                    tokenizer,
                )
            )
            batch_pred_text.extend(microbatch_pred_text)
            batch_unnorm_pred_text.extend(microbatch_unnorm_pred_text)
            batch_tgt_text.extend(microbatch_tgt_text)
            batch_audio_files.extend(audio_files)
            batch_text_files.extend(transcript_files)
            batch_audio_arr.extend(padded_audio_arr)

        # after accumulation_steps, update weights
        if ((batch_idx + 1) % accumulation_steps) == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            # using FSDP clip_grad_norm_ instead of torch.nn.utils.clip_grad_norm_ b/c shard strategy is FULL_SHARD
            # clip_grad_norm_(model.parameters(), max_grad_norm)
            model.clip_grad_norm_(max_norm=max_grad_norm)
            start_optim_step = time.time()
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            end_optim_step = time.time()
            if rank == 0:
                wandb.log(
                    {
                        "efficiency/optim_step_time": end_optim_step - start_optim_step,
                        "global_step": global_step,
                    }
                )
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

            if (global_step % train_log_freq) == 0 or global_step == 1:
                train_loss_tensor = total_loss.clone()
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                train_loss_all = train_loss_tensor.item() / dist.get_world_size()

                if global_step > 1:
                    (
                        norm_tgt_pred_pairs,
                        train_wer_all,
                        train_subs_all,
                        train_dels_all,
                        train_ins_all,
                    ) = calc_pred_wer(batch_tgt_text, batch_pred_text, normalizer)

            if global_step >= epoch_steps + (epoch_steps * epoch):
                if rank == 0 and (global_step % train_log_freq) == 0:
                    print("Logging results at an epoch")
                    print(f"global_step: {global_step}")
                    print(f"train_loss: {train_loss_all}")
                    print(f"train_wer: {train_wer_all}")

                    train_metrics = defaultdict(float)
                    train_metrics["global_step"] = global_step
                    train_metrics["train/train_loss"] = train_loss_all
                    train_metrics["train/train_wer"] = train_wer_all
                    train_metrics["train/train_subs"] = train_subs_all
                    train_metrics["train/train_dels"] = train_dels_all
                    train_metrics["train/train_ins"] = train_ins_all

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
                if rank == 0 and (global_step % train_log_freq) == 0:
                    print("Logging final training results")
                    print(f"global_step: {global_step}")
                    print(f"train_loss: {train_loss_all}")
                    print(f"train_wer: {train_wer_all}")

                    train_metrics = defaultdict(float)
                    train_metrics["global_step"] = global_step
                    train_metrics["train/train_loss"] = train_loss_all
                    train_metrics["train/train_wer"] = train_wer_all
                    train_metrics["train/train_subs"] = train_subs_all
                    train_metrics["train/train_dels"] = train_dels_all
                    train_metrics["train/train_ins"] = train_ins_all

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

            if global_step % ckpt_freq == 0:
                eval_ckpt = save_ckpt(
                    rank=rank,
                    global_step=global_step,
                    local_step=local_step,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    model_dims=model_dims,
                    tags=tags,
                    model_variant=model_variant,
                    exp_name=exp_name,
                    run_id=run_id,
                    file_name="checkpoint",
                    ckpt_dir=ckpt_dir,
                )

            if rank == 0:
                if global_step % train_log_freq == 0 or global_step == 1:
                    print(f"global_step: {global_step}")
                    print(f"train_loss: {train_loss_all}")
                    train_metrics = defaultdict(float)
                    train_metrics["global_step"] = global_step
                    train_metrics["train/train_loss"] = train_loss_all

                    if global_step > 1:
                        print(f"train_wer: {train_wer_all}")
                        train_metrics["train/train_wer"] = train_wer_all
                        train_metrics["train/train_subs"] = train_subs_all
                        train_metrics["train/train_dels"] = train_dels_all
                        train_metrics["train/train_ins"] = train_ins_all

                        print(
                            f"""
                            {len(batch_audio_files)=},
                            {len(batch_audio_arr)=}, 
                            {len(batch_text_files)=}, 
                            {len(batch_pred_text)=}, 
                            {len(batch_tgt_text)=}, 
                            {len(batch_unnorm_pred_text)=}, 
                            {len(norm_tgt_pred_pairs)=}
                            """
                        )

                        log_tbl(
                            global_step=global_step,
                            train_table=train_table,
                            run_id=run_id,
                            batch_audio_files=batch_audio_files,
                            batch_audio_arr=batch_audio_arr,
                            batch_text_files=batch_text_files,
                            batch_pred_text=batch_pred_text,
                            batch_tgt_text=batch_tgt_text,
                            batch_unnorm_pred_text=batch_unnorm_pred_text,
                            norm_tgt_pred_pairs=norm_tgt_pred_pairs,
                        )
                        train_table = wandb.Table(columns=TRAIN_TABLE_COLS)

                    wandb.log(train_metrics)

            # validation
            if run_val:
                if (global_step % val_freq) == 0 and global_step > 0:
                    validate(
                        model=model,
                        precision=precision,
                        tokenizer=tokenizer,
                        normalizer=normalizer,
                        rank=rank,
                        global_step=global_step,
                        batch_size=val_batch_size,
                        num_workers=val_num_workers,
                        n_text_ctx=model_dims.n_text_ctx,
                        cache_dir=val_cache_dir,
                        wandb_log=val_wandb_log,
                    )

                if (global_step % val_freq) == 0 and global_step > 0:
                    print(f"Rank {rank} reaching barrier")
                dist.barrier()
                if (global_step % val_freq) == 0 and global_step > 0:
                    print(f"Rank {rank} passing barrier")

            # evaluation
            if run_eval:
                if (global_step % eval_freq) == 0 and global_step > 0:
                    for eval_set in ["librispeech_clean", "librispeech_other"]:
                        run_async_eval(
                            rank=rank,
                            exp_name=exp_name,
                            eval_script_path=eval_script_path,
                            current_step=global_step,
                            batch_size=eval_batch_size,
                            num_workers=2,
                            ckpt=eval_ckpt,
                            eval_set=eval_set,
                            train_run_id=run_id,
                            log_dir=log_dir,
                            run_id_dir=run_id_dir,
                            eval_dir=eval_dir,
                            wandb_log=eval_wandb_log,
                            cuda=eval_on_gpu,
                        )

                if (global_step % eval_freq) == 0 and global_step > 0:
                    print(f"Rank {rank} reaching barrier")
                dist.barrier()
                if (global_step % eval_freq) == 0 and global_step > 0:
                    print(f"Rank {rank} passing barrier")

        start_dl = time.time()

        batch_pred_text = []
        batch_tgt_text = []
        batch_unnorm_pred_text = []
        batch_audio_files = []
        batch_text_files = []
        batch_audio_arr = []

    # If your dataset size is not a multiple of (batch_size * accumulation_steps)
    # Make sure to account for the last set of batches smaller than accumulation_steps
    if total_loss > 0.0:
        train_loss_tensor = total_loss.clone()
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss_all = train_loss_tensor.item() / dist.get_world_size()

        if scaler is not None:
            scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        global_step += 1
        scheduler.step()

        if global_step >= epoch_steps + (epoch_steps * epoch):
            if rank == 0 and (global_step % train_log_freq) == 0:
                print("Logging results at an epoch")
                print(f"global_step: {global_step}")
                print(f"train_loss: {train_loss_all}")

                train_metrics = defaultdict(float)
                train_metrics["global_step"] = global_step
                train_metrics["train/train_loss"] = train_loss_all

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
            if rank == 0 and (global_step % train_log_freq) == 0:
                print("Logging results at an epoch")
                print(f"global_step: {global_step}")
                print(f"train_loss: {train_loss_all}")

                train_metrics = defaultdict(float)
                train_metrics["global_step"] = global_step
                train_metrics["train/train_loss"] = train_loss_all

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

        current_lr = optimizer.param_groups[0]["lr"]
        if rank == 0:
            wandb.log({"train/learning_rate": current_lr, "global_step": global_step})
        optimizer.zero_grad()
        total_loss = 0.0

        if rank == 0 and (global_step % train_log_freq) == 0:
            print("Logging results at an epoch")
            print(f"global_step: {global_step}")
            print(f"train_loss: {train_loss_all}")

            train_metrics = defaultdict(float)
            train_metrics["global_step"] = global_step
            train_metrics["train/train_loss"] = train_loss_all

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


class ValidationDataset(Dataset):
    def __init__(
        self: str, n_text_ctx: int, val_set: str, hf_token: str, cache_dir: str
    ):
        valset2config = {
            "LIUM/tedlium": {
                "name": "release3",
                "split": "validation",
            },
            "facebook/voxpopuli": {
                "name": "en",
                "split": "validation",
            },
            "mozilla-foundation/common_voice_5_1": {
                "name": "en",
                "split": "validation",
            },
            "distil-whisper/ami-sdm": {
                "name": "sdm",
                "split": "validation",
            },
        }
        self.val_set = val_set
        self.n_text_ctx = n_text_ctx
        self.dataset = load_dataset(
            path=val_set,
            name=valset2config[val_set]["name"],
            split=valset2config[val_set]["split"],
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=hf_token,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        global tokenizer
        if self.val_set == "LIUM/tedlium":
            waveform = self.dataset[index]["audio"]["array"]
            sampling_rate = self.dataset[index]["audio"]["sampling_rate"]
            text_y = self.dataset[index]["text"]
            audio_arr, audio_input = self.preprocess_audio(waveform, sampling_rate)
            text_input, text_y, padding_mask = self.preprocess_text(text_y)
        elif self.val_set == "facebook/voxpopuli":
            waveform = self.dataset[index]["audio"]["array"]
            sampling_rate = self.dataset[index]["audio"]["sampling_rate"]
            text_y = self.dataset[index]["raw_text"]
            audio_arr, audio_input = self.preprocess_audio(waveform, sampling_rate)
            text_input, text_y, padding_mask = self.preprocess_text(text_y)
        elif self.val_set == "mozilla-foundation/common_voice_5_1":
            waveform = self.dataset[index]["audio"]["array"]
            sampling_rate = self.dataset[index]["audio"]["sampling_rate"]
            text_y = self.dataset[index]["sentence"]
            audio_arr, audio_input = self.preprocess_audio(waveform, sampling_rate)
            text_input, text_y, padding_mask = self.preprocess_text(text_y)
        elif self.val_set == "distil-whisper/ami-sdm":
            waveform = self.dataset[index]["audio"]["array"]
            sampling_rate = self.dataset[index]["audio"]["sampling_rate"]
            text_y = self.dataset[index]["text"]
            audio_arr, audio_input = self.preprocess_audio(waveform, sampling_rate)
            text_input, text_y, padding_mask = self.preprocess_text(text_y)

        return audio_arr, audio_input, text_input, text_y, padding_mask

    def preprocess_audio(self, waveform: np.ndarray, sampling_rate: int):
        if sampling_rate != 16000:
            waveform = librosa.resample(
                waveform, orig_sr=sampling_rate, target_sr=16000
            )
        audio_arr = audio.pad_or_trim(waveform)
        audio_arr = audio_arr.astype(np.float32)
        audio_input = audio.log_mel_spectrogram(audio_arr)
        return audio_arr, audio_input

    def preprocess_text(self, text: str):
        if text.strip() == "":
            tokens = (
                list(tokenizer.sot_sequence_including_notimestamps)
                + [tokenizer.no_speech]
                + [tokenizer.eot]
            )
        else:
            tokens = (
                list(tokenizer.sot_sequence_including_notimestamps)
                + tokenizer.encode(text)
                + [tokenizer.eot]
            )

        text_input = tokens[:-1]
        text_y = tokens[1:]

        padding_mask = torch.zeros((self.n_text_ctx, self.n_text_ctx))
        padding_mask[:, len(text_input) :] = -np.inf

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


def validate(
    model: FSDP,
    precision: torch.dtype,
    tokenizer: whisper.tokenizer.Tokenizer,
    normalizer: EnglishTextNormalizer,
    rank: int,
    global_step: int,
    batch_size: int,
    num_workers: int,
    n_text_ctx: int,
    cache_dir: str,
    wandb_log: bool,
):
    val_sets = [
        "LIUM/tedlium",
        "facebook/voxpopuli",
        "mozilla-foundation/common_voice_5_1",
        "distil-whisper/ami-sdm",
    ]
    avg_val_losses = []
    val_table = wandb.Table(columns=VAL_TABLE_COLS)
    all_dataloaders_len = 0
    hf_token = os.getenv("HF_TOKEN")

    for val_set in val_sets:
        dataset = ValidationDataset(
            n_text_ctx=n_text_ctx,
            val_set=val_set,
            hf_token=hf_token,
            cache_dir=cache_dir,
        )
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=sampler,
            worker_init_fn=init_tokenizer,
        )
        model.eval()
        val_loss = 0
        all_dataloaders_len += len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            audio_arr, audio_input, text_input, text_y, padding_mask = batch

            audio_input = audio_input.to(rank)
            text_input = text_input.to(rank)
            text_y = text_y.to(rank)
            padding_mask = padding_mask.to(rank)

            with autocast(device_type="cuda", dtype=precision):
                with torch.no_grad():
                    logits = model(audio_input, text_input, padding_mask)

                    loss = F.cross_entropy(
                        logits.view(-1, logits.shape[-1]),
                        text_y.view(-1),
                        ignore_index=51864,
                    )

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss_all = loss.item() / dist.get_world_size()

            val_loss += loss_all

            if batch_idx // 5 == 1 and wandb_log:
                if rank == 0:
                    pred_text, unnorm_pred_text, tgt_text = gen_pred(
                        logits,
                        text_y,
                        tokenizer,
                    )
                    norm_tgt_pred_pairs, val_wer, val_subs, val_dels, val_ins = (
                        calc_pred_wer(
                            tgt_text,
                            pred_text,
                            normalizer,
                        )
                    )
                    print(f"Validation WER: {val_wer}%")
                    wandb.log(
                        {
                            f"val/{val_set_name}_wer": val_wer,
                            f"val/{val_set_name}_subs": val_subs,
                            f"val/{val_set_name}_dels": val_dels,
                            f"val/{val_set_name}_ins": val_ins,
                            "global_step": global_step,
                        }
                    )

                    for i, (
                        tgt_text_instance,
                        pred_text_instance,
                    ) in enumerate(norm_tgt_pred_pairs):
                        wer = np.round(
                            ow.utils.calculate_wer(
                                (tgt_text_instance, pred_text_instance)
                            ),
                            2,
                        )
                        subs = 0
                        dels = 0
                        ins = 0
                        if len(tgt_text_instance) == 0:
                            subs = 0
                            dels = 0
                            ins = len(pred_text_instance.split())
                        else:
                            measures = jiwer.compute_measures(
                                tgt_text_instance, pred_text_instance
                            )
                            subs = measures["substitutions"]
                            dels = measures["deletions"]
                            ins = measures["insertions"]

                        val_table.add_data(
                            val_set,
                            wandb.Audio(audio_arr[i], sample_rate=16000),
                            pred_text_instance,
                            unnorm_pred_text[i],
                            pred_text[i],
                            tgt_text_instance,
                            tgt_text[i],
                            subs,
                            dels,
                            ins,
                            len(tgt_text_instance.split()),
                            wer,
                        )

                dist.barrier()

        avg_val_loss = val_loss / len(dataloader)
        avg_val_losses.append(avg_val_loss)

        if rank == 0:
            val_set_name = val_set.split("/")[-1]
            print(f"Validation loss for {val_set}: {avg_val_loss}")
            wandb.log(
                {
                    f"val/{val_set_name}_loss": avg_val_loss,
                    "global_step": global_step,
                }
            )

    if rank == 0:
        wandb.log({f"val_table_{global_step}": val_table})
    global_avg_val_loss = sum(avg_val_losses) / len(avg_val_losses)

    if rank == 0:
        print(f"Global average validation loss: {global_avg_val_loss}")
        wandb.log(
            {
                "val/global_avg_loss": global_avg_val_loss,
                "global_step": global_step,
            }
        )


def run_async_eval(
    rank: int,
    exp_name: str,
    eval_script_path: str,
    current_step: int,
    batch_size: int,
    num_workers: int,
    ckpt: str,
    eval_set: Literal[
        "librispeech_clean",
        "librispeech_other",
        "artie_bias_corpus",
        "fleurs",
        "tedlium",
        "voxpopuli",
        "common_voice",
        "ami_ihm",
        "ami_sdm",
    ],
    train_run_id: Optional[str],
    log_dir: str,
    run_id_dir: str,
    eval_dir: str,
    wandb_log: bool = False,
    cuda: bool = True,
) -> None:
    wandb_log_dir = os.getenv("WANDB_DIR")
    hf_token = os.getenv("HF_TOKEN")
    cmd = [
        "python",
        eval_script_path,
        "short_form_eval",
        f"--batch_size={batch_size}",
        f"--num_workers={num_workers}",
        f"--ckpt={ckpt}",
        f"--eval_set={eval_set}",
        f"--log_dir={log_dir}",
        f"--current_step={current_step}",
        f"--train_exp_name={exp_name}",
        f"--train_run_id={train_run_id}",
        f"--wandb_log={wandb_log}",
        f"--wandb_log_dir={wandb_log_dir}",
        f"--run_id_dir={run_id_dir}",
        f"--eval_dir={eval_dir}",
        f"--hf_token={hf_token}",
        f"--cuda={cuda}",
    ]

    print(f"{cmd=}")

    if rank == 0:
        subprocess.Popen(cmd)


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
    ckpt_file_name: Optional[str] = None,
    ckpt_dir: str = "checkpoints",
    log_dir: str = "logs",
    eval_dir: str = "data/eval",
    val_cache_dir: str = "data/val_cache",
    run_id_dir: str = "run_ids",
    lr: float = 1.5e-3,
    betas: tuple = (0.9, 0.98),
    eps: float = 1e-6,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    eff_batch_size: int = 256,
    train_batch_size: int = 8,
    eval_batch_size: Optional[int] = 32,
    val_batch_size: int = 32,
    num_workers: int = 10,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    shuffle: bool = True,
    persistent_workers: bool = True,
    run_val: bool = True,
    val_freq: Optional[int] = 20000,
    val_wandb_log: bool = True,
    run_eval: bool = False,
    train_log_freq: int = 20000,
    eval_freq: Optional[int] = 20000,
    ckpt_freq: int = 2500,
    verbose: bool = False,
    precision: Literal["bfloat16", "float16", "pure_float16", "float32"] = "bfloat16",
    hardware: str = "H100",
    eval_script_path: str = "eval.py",
    eval_wandb_log: bool = False,
    eval_on_gpu: bool = True,
    sharding_strategy: Literal[
        "FULL_SHARD",
        "SHARD_GRAD_OP",
        "HYBRID_SHARD",
        "_HYBRID_SHARD_ZERO2",
    ] = "FULL_SHARD",
) -> None:
    """Main function for training

    Conducts a training loop for the specified number of steps, with validation and evaluation (if run_eval is True)

    Args:
        model_variant: The variant of the model to use
        exp_name: The name of the experiment
        job_type: The type of job (e.g., training, evaluation)
        filter: The filter to use for the dataset
        run_id: The run ID to use for loading a checkpoint
        rank: The rank of the current process
        world_size: The total number of processes
        lr: The learning rate
        betas: The betas for the optimizer
        eps: The epsilon for the optimizer
        weight_decay: The weight decay for the optimizer
        max_grad_norm: The maximum gradient norm
        subset: The subset of the dataset to use
        eff_size: The size of the efficientnet model
        train_batch_size: The batch size for training
        val_batch_size: The batch size for validation
        eval_batch_size: The batch size for evaluation
        train_val_split: The train-validation split
        num_workers: The number of workers for the dataloader
        pin_memory: Whether to pin memory for the dataloader
        shuffle: Whether to shuffle the dataloader
        persistent_workers: Whether to use persistent workers for the dataloader
        run_eval: Whether to run evaluation
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(run_id_dir):
        os.makedirs(run_id_dir, exist_ok=True)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    if not os.path.exists(f"{run_id_dir}/{exp_name}.txt"):
        run_id = None
    else:
        with open(f"{run_id_dir}/{exp_name}.txt", "r") as f:
            run_id = f.read().strip()

        # in the case that previous job crashed before a ckpt could be saved, generate a new run_id
        if not os.path.exists(f"{ckpt_dir}/{exp_name}_{run_id}"):
            run_id = None

    tags = [
        "fsdp-train",
        "grad-acc",
        "bfloat16",
    ]

    if ckpt_file_name is None:
        ckpt_file_name = ""

    model_dims = VARIANT_TO_DIMS[model_variant]

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    # setup the process groups
    setup(local_rank)

    if rank == 0:
        print(
            f"""
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
    samples_dicts_files = glob.glob(f"{samples_dicts_dir}/*.jsonl.*")
    print(f"{len(samples_dicts_files)=}")

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
    if run_id is not None or "/" in ckpt_file_name:
        (
            global_step,
            local_step,
            epoch,
            model,
            optimizer,
            scaler,
            scheduler,
            accumulation_steps,
            warmup_steps,
            train_steps,
        ) = load_ckpt(
            exp_name=exp_name,
            run_id=run_id,
            rank=local_rank,
            world_size=world_size,
            train_steps=train_steps,
            train_batch_size=train_batch_size,
            eff_batch_size=eff_batch_size,
            file_name=ckpt_file_name,
            ckpt_dir=ckpt_dir,
            model_variant=model_variant,
            precision=precision,
            precision_policy=precision_policy,
            sharding_strategy=sharding_strategy,
        )
    else:
        model = ow.model.Whisper(dims=model_dims).to(local_rank)

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                ResidualAttentionBlock,
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

        with open(f"{run_id_dir}/{exp_name}.txt", "w") as f:
            f.write(run_id)

        os.makedirs(f"{log_dir}/training/{exp_name}/{run_id}", exist_ok=True)

    # for other ranks, need to access file for run_id
    dist.barrier()  # wait for rank 0 to write run_id to file and then read it
    if rank != 0:
        with open(f"{run_id_dir}/{exp_name}.txt", "r") as f:
            run_id = f.read().strip()

    while global_step < train_steps:
        (
            global_step,
            local_step,
            epoch,
            model,
            optimizer,
            scaler,
            scheduler,
        ) = train(
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
            tokenizer=tokenizer,
            normalizer=normalizer,
            optimizer=optimizer,
            scheduler=scheduler,
            accumulation_steps=accumulation_steps,
            max_grad_norm=max_grad_norm,
            model_dims=model_dims,
            model_variant=model_variant,
            run_val=run_val,
            val_freq=val_freq,
            val_batch_size=val_batch_size,
            val_num_workers=num_workers,
            val_cache_dir=val_cache_dir,
            val_wandb_log=val_wandb_log,
            run_eval=run_eval,
            run_id=run_id,
            tags=tags,
            exp_name=exp_name,
            log_dir=log_dir,
            ckpt_dir=ckpt_dir,
            train_log_freq=train_log_freq,
            eval_freq=eval_freq,
            ckpt_freq=ckpt_freq,
            verbose=verbose,
            precision=autocast_precision,
            eval_script_path=eval_script_path,
            eval_dir=eval_dir,
            eval_wandb_log=eval_wandb_log,
            eval_batch_size=eval_batch_size,
            run_id_dir=run_id_dir,
            eval_on_gpu=eval_on_gpu,
        )

        epoch += 1

        eval_ckpt = save_ckpt(
            rank=rank,
            global_step=global_step,
            local_step=local_step,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            model_dims=model_dims,
            tags=tags,
            model_variant=model_variant,
            exp_name=exp_name,
            run_id=run_id,
            file_name="latesttrain",
            ckpt_dir=ckpt_dir,
        )

        if run_eval:
            print(f"Evaluation after epoch at {global_step=} on rank {rank}")
            for eval_set in ["librispeech_clean", "librispeech_other"]:
                run_async_eval(
                    rank=rank,
                    exp_name=exp_name,
                    eval_script_path=eval_script_path,
                    current_step=global_step,
                    batch_size=eval_batch_size,
                    num_workers=2,
                    ckpt=eval_ckpt,
                    eval_set=eval_set,
                    train_run_id=run_id,
                    log_dir=log_dir,
                    run_id_dir=run_id_dir,
                    eval_dir=eval_dir,
                    wandb_log=eval_wandb_log,
                    cuda=eval_on_gpu,
                )
            print(f"Rank {rank} reaching barrier")
            dist.barrier()
            print(f"Rank {rank} passing barrier")

    cleanup()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    Fire(main)
