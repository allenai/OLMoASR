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
import gzip
import subprocess

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast

import whisper
from whisper import audio, DecodingOptions
from whisper.normalizers import EnglishTextNormalizer
from whisper.tokenizer import get_tokenizer
import whisper.tokenizer
from open_whisper.config.model_dims import VARIANT_TO_DIMS, ModelDimensions
import open_whisper as ow

from scripts.eval.eval import EvalDataset
from for_logging import TRAIN_TABLE_COLS, EVAL_TABLE_COLS

WANDB_EXAMPLES = 8
os.environ["WANDB__SERVICE_WAIT"] = "300"
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

        text_tokens = list(tokenizer.sot_sequence_including_notimestamps) + text_tokens + [tokenizer.eot]

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


def save_ckpt(
    global_step: int,
    local_step: int,
    epoch: int,
    best_eval_wer: Optional[float],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
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
        global_step: The current step
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
    ddp_checkpoint = {
        "global_step": global_step,
        "local_step": local_step,
        "epoch": epoch,
        "best_eval_wer": best_eval_wer,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        # You can also save other items such as scheduler state
        "scheduler_state_dict": (scheduler.state_dict() if scheduler else None),
        "dims": model_dims,
        # Include any other information you deem necessary
    }

    non_ddp_checkpoint = {
        "global_step": global_step,
        "local_step": local_step,
        "epoch": epoch,
        "best_eval_wer": best_eval_wer,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        # You can also save other items such as scheduler state
        "scheduler_state_dict": (scheduler.state_dict() if scheduler else None),
        "dims": model_dims,
    }

    os.makedirs(f"{ckpt_dir}/{exp_name}_{run_id}", exist_ok=True)

    if file_name != "latesttrain":
        if len(glob.glob(f"{ckpt_dir}/{exp_name}_{run_id}/{file_name}_*.pt")) > 0:
            for p in glob.glob(f"{ckpt_dir}/{exp_name}_{run_id}/{file_name}_*.pt"):
                if "inf" not in p:
                    os.remove(p)

    torch.save(
        ddp_checkpoint,
        f"{ckpt_dir}/{exp_name}_{run_id}/{file_name}_{global_step:08}_{model_variant}_{'_'.join(tags)}_ddp.pt",
    )
    torch.save(
        non_ddp_checkpoint,
        f"{ckpt_dir}/{exp_name}_{run_id}/{file_name}_{global_step:08}_{model_variant}_{'_'.join(tags)}_non_ddp.pt",
    )


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
    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}

    if file_name is "":
        all_ckpt_files = glob.glob(
            f"{ckpt_dir}/{exp_name}_{run_id}/*_{model_variant}_*_fp16_ddp.pt"
        )
        latest_step = max([int(f.split("/")[-1].split("_")[1]) for f in all_ckpt_files])
        ckpt_file = glob.glob(
            f"{ckpt_dir}/{exp_name}_{run_id}/*_{latest_step:08}_{model_variant}_*_fp16_ddp.pt"
        )[0]
        print(f"{ckpt_file=}")
        # latest_ckpt_file = max(all_ckpt_files, key=os.path.getctime)
    elif "/" in file_name:
        ckpt_file = file_name
    else:
        ckpt_file = glob.glob(
            f"{ckpt_dir}/{exp_name}_{run_id}/{file_name}_*_{model_variant}_*_fp16_ddp.pt"
        )[0]
        print(f"{ckpt_file=}")

    ckpt = torch.load(ckpt_file, map_location=map_location, weights_only=False)

    model = ow.model.Whisper(dims=ckpt["dims"]).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # if end at training step i, then start at step i+1 when resuming
    global_step = ckpt["global_step"]
    local_step = ckpt["local_step"]

    epoch = ckpt["epoch"]

    best_eval_wer = ckpt["best_eval_wer"]

    model.load_state_dict(ckpt["model_state_dict"])

    optimizer = AdamW(model.parameters())
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    scaler = GradScaler()
    scaler.load_state_dict(ckpt["scaler_state_dict"])

    scheduler, accumulation_steps, warmup_steps, train_steps = prepare_sched(
        train_steps=train_steps,
        world_size=world_size,
        train_batch_size=train_batch_size,
        eff_batch_size=eff_batch_size,
        optimizer=optimizer,
    )
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return (
        global_step,
        local_step,
        epoch,
        best_eval_wer,
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
        pred_instance_text = tokenizer.decode(list(pred_instance))
        microbatch_unnorm_pred_text.append(pred_instance_text)
        pred_instance_text = ow.utils.remove_after_endoftext(pred_instance_text)
        microbatch_pred_text.append(pred_instance_text)

    microbatch_tgt_text = []
    for text_y_instance in text_y.cpu().numpy():
        tgt_y_instance_text = tokenizer.decode(list(text_y_instance))
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
    scaler: GradScaler,
    model: DDP,
    tokenizer: whisper.tokenizer.Tokenizer,
    normalizer: EnglishTextNormalizer,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    accumulation_steps: int,
    max_grad_norm: float,
    model_dims: Optional[ModelDimensions],
    model_variant: Optional[str],
    best_eval_wer: Optional[float],
    run_eval: bool,
    eval_loaders: Optional[List[DataLoader]],
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
    async_eval: bool,
    eval_script_path: Optional[str],
    eval_dir: Optional[str],
    eval_wandb_log: bool,
    eval_batch_size: Optional[int],
    run_id_dir: Optional[str],
    eval_on_gpu: bool,
) -> Tuple[
    int,
    float,
    torch.nn.Module,
    torch.optim.Optimizer,
    GradScaler,
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
                    "efficiency/data_to_gpu_time": end_data_to_gpu - start_data_to_gpu,
                    "efficiency/fwd_time": end_fwd - start_fwd,
                    "efficiency/avg_preproc_time": sum(preproc_time)
                    / len(preproc_time),
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
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)
            start_optim_step = time.time()
            scaler.step(optimizer)
            end_optim_step = time.time()
            if rank == 0:
                wandb.log(
                    {
                        "efficiency/optim_step_time": end_optim_step - start_optim_step,
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
                    best_eval_wer,
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
                    best_eval_wer,
                    model,
                    optimizer,
                    scaler,
                    scheduler,
                )

            optimizer.zero_grad()  # Reset gradients only after updating weights
            total_loss = 0.0

            if rank == 0:
                if global_step % ckpt_freq == 0:
                    save_ckpt(
                        global_step=global_step,
                        local_step=local_step,
                        epoch=epoch,
                        best_eval_wer=best_eval_wer,
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

            # evaluation
            if run_eval:
                if (global_step % eval_freq) == 0 and global_step > 0:
                    if async_eval == True:
                        for eval_set in ["librispeech_clean", "librispeech_other"]:
                            run_async_eval(
                                rank=rank,
                                exp_name=exp_name,
                                eval_script_path=eval_script_path,
                                current_step=global_step,
                                batch_size=eval_batch_size,
                                num_workers=2,
                                ckpt=f"{ckpt_dir}/{exp_name}_{run_id}/checkpoint_{global_step:08}_{model_variant}_{'_'.join(tags)}_non_ddp.pt",
                                eval_set=eval_set,
                                train_run_id=run_id,
                                log_dir=log_dir,
                                run_id_dir=run_id_dir,
                                eval_dir=eval_dir,
                                wandb_log=eval_wandb_log,
                                cuda=eval_on_gpu,
                            )
                    else:
                        best_eval_wer = evaluate(
                            rank=rank,
                            local_rank=local_rank,
                            global_step=global_step,
                            local_step=local_step,
                            epoch=epoch,
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            scheduler=scheduler,
                            model_dims=model_dims,
                            model_variant=model_variant,
                            eval_loaders=eval_loaders,
                            normalizer=normalizer,
                            best_eval_wer=best_eval_wer,
                            tags=tags,
                            exp_name=exp_name,
                            run_id=run_id,
                            table_idx=None,
                            log_dir=log_dir,
                            ckpt_dir=ckpt_dir,
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

        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
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
                best_eval_wer,
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
                best_eval_wer,
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
        best_eval_wer,
        model,
        optimizer,
        scaler,
        scheduler,
    )


def evaluate(
    rank: int,
    local_rank: int,
    global_step: int,
    local_step: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: LambdaLR,
    model_dims: ModelDimensions,
    model_variant: str,
    eval_loaders: List[DataLoader],
    normalizer: EnglishTextNormalizer,
    best_eval_wer: Optional[float],
    tags: Optional[List[str]],
    exp_name: Optional[str],
    run_id: Optional[str],
    table_idx: Optional[str],
    log_dir: str,
    ckpt_dir: str,
) -> None:
    """Evaluation loop for 1 epoch

    Evaluation loop with WER calculation for 2 corpora: librispeech-clean and librispeech-other

    Args:
        rank: The rank of the current process
        eval_batch_size: The batch size for evaluation
        num_workers: The number of workers for the dataloader
        model: The model to evaluate
        normalizer: The text normalizer
        tags: The tags to use for logging
        exp_name: The experiment name
    """
    eval_table = wandb.Table(columns=EVAL_TABLE_COLS)

    non_ddp_model = model.module
    non_ddp_model.eval()

    eval_wers = []

    for eval_set, eval_dataloader in eval_loaders:
        print(f"Evaluating {eval_set}\n")

        hypotheses = []
        references = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(eval_dataloader), total=len(eval_dataloader)
            ):
                audio_fp, _, audio_input, text_y = batch
                audio_input = audio_input.to(local_rank)

                options = DecodingOptions(language="en", without_timestamps=True)

                results = non_ddp_model.decode(audio_input, options=options)
                pred_text = [result.text for result in results]
                norm_pred_text = [normalizer(text) for text in pred_text]
                hypotheses.extend(norm_pred_text)
                norm_tgt_text = [normalizer(text) for text in text_y]
                references.extend(norm_tgt_text)

                if rank == 0:
                    if (batch_idx + 1) % int(np.ceil(len(eval_dataloader) / 10)) == 0:
                        for i in range(0, len(pred_text), 16):
                            wer = (
                                np.round(
                                    jiwer.wer(
                                        reference=norm_tgt_text[i],
                                        hypothesis=norm_pred_text[i],
                                    ),
                                    2,
                                )
                                * 100
                            )
                            measures = jiwer.compute_measures(
                                truth=norm_tgt_text[i], hypothesis=norm_pred_text[i]
                            )
                            subs = measures["substitutions"]
                            dels = measures["deletions"]
                            ins = measures["insertions"]

                            eval_table.add_data(
                                run_id,
                                eval_set,
                                wandb.Audio(audio_fp[i], sample_rate=16000),
                                pred_text[i],
                                norm_pred_text[i],
                                norm_tgt_text[i],
                                subs,
                                dels,
                                ins,
                                wer,
                            )

                        wer = (
                            jiwer.wer(
                                reference=norm_tgt_text, hypothesis=norm_pred_text
                            )
                            * 100
                        )

            avg_wer = jiwer.wer(references, hypotheses) * 100
            eval_wers.append(avg_wer)

            if rank == 0:
                with open(
                    f"{log_dir}/training/{exp_name}/{run_id}/eval_results_{'_'.join(tags)}.txt",
                    "a",
                ) as f:
                    f.write(
                        f"{eval_set} average WER: {avg_wer}\n at step {global_step}\n"
                    )
                wandb.log({f"eval/{eval_set}_wer": avg_wer, "global_step": global_step})

    if rank == 0:
        if table_idx is not None:
            wandb.log({f"eval_table_{table_idx}": eval_table})
        else:
            wandb.log({f"eval_table_{global_step}": eval_table})

        avg_eval_wer = np.mean(eval_wers)

        if avg_eval_wer < best_eval_wer:
            best_eval_wer = avg_eval_wer
            print("Saving best eval model")
            save_ckpt(
                global_step=global_step,
                local_step=local_step,
                epoch=epoch,
                best_eval_wer=best_eval_wer,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                model_dims=model_dims,
                tags=tags,
                model_variant=model_variant,
                exp_name=exp_name,
                run_id=run_id,
                file_name="besteval",
                ckpt_dir=ckpt_dir,
            )

    return best_eval_wer


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
    run_id_dir: str = "run_ids",
    lr: float = 1.5e-3,
    betas: tuple = (0.9, 0.98),
    eps: float = 1e-6,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    eff_batch_size: int = 256,
    train_batch_size: int = 8,
    eval_batch_size: Optional[int] = 32,
    num_workers: int = 10,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    shuffle: bool = True,
    persistent_workers: bool = True,
    run_eval: bool = False,
    train_log_freq: int = 20000,
    eval_freq: Optional[int] = 20000,
    ckpt_freq: int = 2500,
    verbose: bool = False,
    precision: ["bfloat16", "float16", "float32"] = "float16",
    hardware: str = "H100",
    async_eval: bool = False,
    eval_script_path: Optional[str] = None,
    eval_wandb_log: bool = False,
    eval_on_gpu: bool = True,
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

    if ckpt_file_name is None:
        ckpt_file_name = ""

    tags = [
        "ddp-train",
        "grad-acc",
        "fp16",
    ]

    model_dims = VARIANT_TO_DIMS[model_variant]
    precision_dict = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    setup(rank=local_rank)

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

    tokenizer = get_tokenizer(multilingual=False)
    normalizer = EnglishTextNormalizer()
    n_text_ctx = model_dims.n_text_ctx
    n_head = model_dims.n_text_head

    samples_dicts_files = glob.glob(f"{samples_dicts_dir}/*.jsonl.gz")
    print(f"{len(samples_dicts_files)=}")

    with multiprocessing.Pool() as pool:
        samples_dicts = list(
            chain(
                *tqdm(
                    pool.imap_unordered(open_dicts_file, samples_dicts_files),
                    total=len(samples_dicts_files),
                )
            )
        )
    print(f"{len(samples_dicts)=}")

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

    if not async_eval and run_eval:
        print(f"Preparing eval sets on rank {rank}")
        eval_sets = ["librispeech_clean", "librispeech_other"]
        eval_loaders = []
        for eval_set in eval_sets:
            eval_dataset = EvalDataset(eval_set=eval_set, eval_dir=eval_dir)

            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                persistent_workers=persistent_workers,
                pin_memory=pin_memory,
            )
            eval_loaders.append((eval_set, eval_dataloader))
    else:
        eval_loaders = None

    # model instantiation
    if run_id is not None or "/" in ckpt_file_name:
        (
            global_step,
            local_step,
            epoch,
            best_eval_wer,
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
        )
    else:
        model = ow.model.Whisper(dims=model_dims).to(local_rank)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

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

        scaler = GradScaler()

        global_step = 0
        local_step = 0
        epoch = 0

        if run_eval:
            best_eval_wer = float("inf")
        else:
            best_eval_wer = None

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

    while global_step < train_steps:
        (
            global_step,
            local_step,
            epoch,
            best_eval_wer,
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
            best_eval_wer=best_eval_wer,
            run_eval=run_eval,
            eval_loaders=eval_loaders,
            run_id=run_id,
            tags=tags,
            exp_name=exp_name,
            log_dir=log_dir,
            ckpt_dir=ckpt_dir,
            train_log_freq=train_log_freq,
            eval_freq=eval_freq,
            ckpt_freq=ckpt_freq,
            verbose=verbose,
            precision=precision_dict[precision],
            async_eval=async_eval,
            eval_script_path=eval_script_path,
            eval_dir=eval_dir,
            eval_wandb_log=eval_wandb_log,
            eval_batch_size=eval_batch_size,
            run_id_dir=run_id_dir,
            eval_on_gpu=eval_on_gpu,
        )

        epoch += 1

        if rank == 0:
            save_ckpt(
                global_step=global_step,
                local_step=local_step,
                epoch=epoch,
                best_eval_wer=best_eval_wer,
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
            if async_eval is True:
                for eval_set in ["librispeech_clean", "librispeech_other"]:
                    run_async_eval(
                        rank=rank,
                        exp_name=exp_name,
                        eval_script_path=eval_script_path,
                        current_step=global_step,
                        batch_size=eval_batch_size,
                        num_workers=2,
                        ckpt=f"{ckpt_dir}/{exp_name}_{run_id}/latesttrain_{global_step:08}_{model_variant}_{'_'.join(tags)}_non_ddp.pt",
                        eval_set=eval_set,
                        train_run_id=run_id,
                        log_dir=log_dir,
                        run_id_dir=run_id_dir,
                        eval_dir=eval_dir,
                        wandb_log=eval_wandb_log,
                        cuda=eval_on_gpu,
                    )
            else:
                best_eval_wer = evaluate(
                    rank=rank,
                    local_rank=local_rank,
                    global_step=global_step,
                    local_step=local_step,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    model_dims=model_dims,
                    model_variant=model_variant,
                    eval_loaders=eval_loaders,
                    normalizer=normalizer,
                    best_eval_wer=best_eval_wer,
                    tags=tags,
                    exp_name=exp_name,
                    run_id=run_id,
                    table_idx=f"epoch_{epoch}",
                    log_dir=log_dir,
                    ckpt_dir=ckpt_dir,
                )

            print(f"Rank {rank} reaching barrier")
            dist.barrier()
            print(f"Rank {rank} passing barrier")

    cleanup()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    Fire(main)
