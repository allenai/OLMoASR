import os
import re
import glob
import numpy as np
import wandb
from typing import List, Tuple, Union, Optional, Literal, Dict
import time
import jiwer
from fire import Fire
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

import whisper
from whisper import audio, DecodingOptions
from whisper.normalizers import EnglishTextNormalizer
from whisper.tokenizer import get_tokenizer
import whisper.tokenizer
from open_whisper.config.model_dims import VARIANT_TO_DIMS, ModelDimensions
import open_whisper as ow

from scripts.eval.eval import EvalDataset
from scripts.training import for_logging

import webdataset as wds
import tempfile

WANDB_EXAMPLES = 8
os.environ["WANDB__SERVICE_WAIT"] = "300"

def bytes_to_file(data_bytes: bytes, suffix: str) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(data_bytes)
    temp_file.flush()
    temp_file.close()

    return temp_file.name

def decode_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    audio_file = bytes_to_file(audio_bytes, ".m4a")
    audio_arr = audio.load_audio(audio_file)
    os.remove(audio_file)

    return audio_arr

def decode_text_bytes(text_bytes: bytes) -> str:
    transcript_str = text_bytes.decode("utf-8")
    transcript_file = bytes_to_file(text_bytes, ".srt")

    return transcript_file

def decode_sample(sample: Dict[str, bytes]) -> Tuple[np.ndarray, str]:
    file_path = os.path.join(sample["__url__"], sample["__key__"])
    audio_path = file_path + ".m4a"
    text_path = file_path + ".srt"
    audio_bytes = sample["m4a"]
    text_bytes = sample["srt"]
    audio_arr = decode_audio_bytes(audio_bytes)
    transcript_file = decode_text_bytes(text_bytes)

    return audio_path, audio_arr, text_path, transcript_file

def preprocess_audio(audio_arr: np.ndarray) -> torch.Tensor:
    audio_arr = audio.pad_or_trim(audio_arr)
    mel_spec = audio.log_mel_spectrogram(audio_arr)

    return mel_spec, audio_arr

def preprocess_text(transcript_file: str, tokenizer: whisper.tokenizer.Tokenizer, n_text_ctx: int) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    reader = ow.utils.TranscriptReader(file_path=transcript_file)
    transcript, *_ = reader.read()
    os.remove(transcript_file)
    
    if not transcript:
        text_tokens = [tokenizer.no_speech]
    else:
        transcript_text = reader.extract_text(transcript=transcript)

        text_tokens = tokenizer.encode(transcript_text)

    text_tokens = list(tokenizer.sot_sequence_including_notimestamps) + text_tokens

    text_tokens.append(tokenizer.eot)

    # offset
    text_input = text_tokens[:-1]
    text_y = text_tokens[1:]

    padding_mask = torch.zeros((n_text_ctx, n_text_ctx))
    padding_mask[:, len(text_input) :] = -float("inf")

    text_input = np.pad(
        text_input,
        pad_width=(0, n_text_ctx - len(text_input)),
        mode="constant",
        constant_values=51864,
    )
    text_y = np.pad(
        text_y,
        pad_width=(0, n_text_ctx - len(text_y)),
        mode="constant",
        constant_values=51864,
    )

    text_input = torch.tensor(text_input, dtype=torch.long)
    text_y = torch.tensor(text_y, dtype=torch.long)

    return text_input, text_y, padding_mask
    
def preprocess(sample, n_text_ctx: int):
    tokenizer = get_tokenizer(multilingual=False)
    audio_path, audio_arr, text_path, transcript_file = decode_sample(sample)
    audio_input, audio_arr = preprocess_audio(audio_arr)
    text_input, text_y, padding_mask = preprocess_text(transcript_file, tokenizer, n_text_ctx)

    return audio_path, text_path, audio_arr, audio_input, text_input, text_y, padding_mask


def setup(rank: int, world_size: int) -> None:
    """Initializes the distributed process group

    Args:
        rank: The rank of the current process
        world_size: The total number of processes
    """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare_data(
    rank: int,
    world_size: int,
    train_shards: str,
    val_shards: str,
    len_train_data: Optional[int],
    len_val_data: Optional[int],
    train_batch_size: int,
    val_batch_size: int,
    n_text_ctx: int,
    pin_memory: bool = True,
    num_workers: int = 0,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, Optional[int], DataLoader, Optional[int]]:
    """Prepares the data for training

    Given the list of audio and transcript files, prepares the webdataset dataloader for training and validation

    Args:
        rank: The rank of the current process
        world_size: The total number of processes
        train_shards: The shards for training (in brace notation)
        val_shards: The shards for validation (in brace notation)
        train_batch_size: The batch size for training
        val_batch_size: The batch size for validation
        n_text_ctx: The number of text tokens
        pin_memory: Whether to pin memory
        num_workers: The number of workers
        persistent_workers: Whether to use persistent workers

    Returns:
        A tuple containing the webdataset dataloader for training and validation
    """
    with open("logs/data/preprocess/num_files.txt", "r") as f:
        shard_to_size = {int(line.split(":")[0]): int(line.split(":")[1]) for line in f.readlines()}
    
    start_train_shard, end_train_shard = [int(shard_idx) for shard_idx in train_shards.split("{")[-1].split("}")[0].split("..")]
    start_val_shard, end_val_shard = [int(shard_idx) for shard_idx in val_shards.split("{")[-1].split("}")[0].split("..")]

    if not len_train_data:
        len_train_data = sum([shard_to_size[shard_idx] for shard_idx in range(start_train_shard, end_train_shard + 1)])
    if not len_val_data:
        len_val_data = sum([shard_to_size[shard_idx] for shard_idx in range(start_val_shard, end_val_shard + 1)])
    
    if len_train_data % train_batch_size == 0:
        len_train_loader = len_train_data // train_batch_size
    else:
        len_train_loader = (len_train_data // train_batch_size) + 1
    
    if len_val_data % val_batch_size == 0:
        len_val_loader = len_val_data // val_batch_size
    else:
        len_val_loader = (len_val_data // val_batch_size) + 1
    
    train_dataset = wds.WebDataset(train_shards, resampled=True).map(lambda sample: preprocess(sample, 448))
    val_dataset = wds.WebDataset(val_shards, resampled=True).map(lambda sample: preprocess(sample, 448))

    # prepare the dataloaders
    train_dataset = train_dataset.batched(train_batch_size).with_length(len_train_data)
    val_dataset = val_dataset.batched(val_batch_size).with_length(len_val_data)

    train_dataloader = wds.WebLoader(train_dataset, batch_size=None, shuffle=False, num_workers=num_workers).unbatched().with_epoch(len_train_loader).batched(train_batch_size)
    val_dataloader = wds.WebLoader(val_dataset, batch_size=None, shuffle=False, num_workers=num_workers).unbatched().with_epoch(len_val_loader).batched(val_batch_size)
    
    return train_dataloader, len_train_data, val_dataloader, len_val_data


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
    len_train_data: int,
    world_size: int,
    train_batch_size: int,
    eff_size: int,
    epochs: int,
    optimizer: torch.optim.Optimizer,
) -> Tuple[LambdaLR, int, int, int]:
    """Prepares the scheduler for training

    Prepares the LambdaLR scheduler for training

    Args:
        len_train_data: The length of the training data
        world_size: The total number of processes
        train_batch_size: The batch size for training
        eff_size: The effective size
        epochs: The number of epochs
        optimizer: The optimizer for training

    Returns:
        A tuple containing the scheduler, the number of steps over which to accumulate gradients, the number of warmup steps, and the total number of steps
    """
    if eff_size <= (world_size * train_batch_size):
        accumulation_steps = 1
    else:
        accumulation_steps = eff_size // (
            world_size * train_batch_size
        )  # Number of steps over which to accumulate gradients

    if len_train_data % train_batch_size == 0:
        len_train_loader = len_train_data // train_batch_size
    else:
        len_train_loader = (len_train_data // train_batch_size) + 1

    total_steps = int(np.ceil(len_train_loader / accumulation_steps) * epochs)
    warmup_steps = np.ceil(0.002 * total_steps)

    def lr_lambda(batch_idx: int) -> float:
        if batch_idx < warmup_steps:
            return float(batch_idx) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - batch_idx) / float(max(1, total_steps - warmup_steps)),
        )

    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

    return scheduler, accumulation_steps, warmup_steps, total_steps


def setup_wandb(
    run_id: Optional[str],
    exp_name: str,
    job_type: str,
    model_variant: str,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    len_train_data: float,
    len_val_data: float,
    train_batch_size: int,
    val_batch_size: int,
    epochs: int,
    total_steps: int,
    warmup_steps: int,
    accumulation_steps: int,
    world_size: int,
    num_workers: int,
) -> Tuple[Optional[str], List[str], wandb.Artifact, wandb.Artifact, bool, bool]:
    """Sets up the Weights and Biases logging

    Args:
        run_id: The run ID
        exp_name: The experiment name
        job_type: The type of job
        model_variant: The variant of the model
        lr: The learning rate
        betas: The betas for the Adam optimizer
        eps: The epsilon value
        weight_decay: The weight decay
        train_batch_size: The batch size for training
        epochs: The number of epochs
        total_steps: The total number of steps
        warmup_steps: The number of warmup steps
        accumulation_steps: The number of steps over which to accumulate gradients
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
        "len_train_data": len_train_data,
        "len_val_data": len_val_data,
        "train_batch_size": train_batch_size,
        "val_batch_size": val_batch_size,
        "epochs": epochs,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "accumulation_steps": accumulation_steps,
        "world_size": world_size,
        "num_workers": num_workers,
        "model_variant": model_variant,
    }

    tags = [
        "ddp-train",
        "grad-acc",
        "fp16",
    ]

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
        tags=(tags),
        name=exp_name,
        dir="logs/training",
    )

    train_res = wandb.Artifact("train_res", type="results")
    train_res_added = False
    val_res = wandb.Artifact("val_res", type="results")
    val_res_added = False

    return run_id, tags, train_res, val_res, train_res_added, val_res_added


def save_ckpt(
    epoch: int,
    best_val_loss: float,
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
) -> None:
    """Save model (DDP) checkpoint

    Saves non-DDP and DDP model checkpoints to checkpoints/{exp_name}_{run_id} directory in the format of {file_name}_{epoch}_{model_variant}_{tags}_{ddp}.pt

    Args:
        epoch: The current epoch number
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
    """
    ddp_checkpoint = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        # You can also save other items such as scheduler state
        "scheduler_state_dict": (scheduler.state_dict() if scheduler else None),
        "dims": model_dims,
        # Include any other information you deem necessary
    }

    non_ddp_checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        # You can also save other items such as scheduler state
        "scheduler_state_dict": (scheduler.state_dict() if scheduler else None),
        "dims": model_dims,
    }

    if epoch != 0:
        prev_epoch = epoch - 1
        os.remove(
            f"checkpoints/{exp_name}_{run_id}/{file_name}_epoch={prev_epoch}_{model_variant}_{'_'.join(tags)}_ddp.pt"
        )
        os.remove(
            f"checkpoints/{exp_name}_{run_id}/{file_name}_epoch={prev_epoch}_{model_variant}_{'_'.join(tags)}_non_ddp.pt"
        )

    os.makedirs(f"checkpoints/{exp_name}_{run_id}", exist_ok=True)

    torch.save(
        ddp_checkpoint,
        f"checkpoints/{exp_name}_{run_id}/{file_name}_{epoch=}_{model_variant}_{'_'.join(tags)}_ddp.pt",
    )
    torch.save(
        non_ddp_checkpoint,
        f"checkpoints/{exp_name}_{run_id}/{file_name}_{epoch=}_{model_variant}_{'_'.join(tags)}_non_ddp.pt",
    )


def load_ckpt(
    exp_name: str,
    run_id: str,
    rank: int,
    world_size: int,
    epochs: int,
    train_batch_size: int,
    eff_size: int,
    len_train_data: int,
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
        epochs: The number of epochs
        train_batch_size: The batch size for training.
        eff_size: The effective size
        train_dataloader: The training dataloader

    Returns:
        A tuple containing the current epoch, the best validation loss, the model, the optimizer, the gradient scaler,
        the scheduler, the number of steps over which to accumulate gradients, the number of warmup steps, and the total number of steps
    """
    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}

    ckpt_file = glob.glob(
        f"checkpoints/{exp_name}_{run_id}/latest_train_*_fp16_ddp.pt"
    )[0]

    ckpt = torch.load(ckpt_file, map_location=map_location)

    model = ow.model.Whisper(dims=ckpt["dims"]).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # if end at training epoch i, then start at epoch i+1 when resuming
    current_epoch = ckpt["epoch"] + 1

    best_val_loss = ckpt["best_val_loss"]

    model.load_state_dict(ckpt["model_state_dict"])

    optimizer = AdamW(model.parameters())
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    scaler = GradScaler()
    scaler.load_state_dict(ckpt["scaler_state_dict"])

    scheduler, accumulation_steps, warmup_steps, total_steps = prepare_sched(
        len_train_data=len_train_data,
        world_size=world_size,
        train_batch_size=train_batch_size,
        eff_size=eff_size,
        epochs=epochs,
        optimizer=optimizer,
    )
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return (
        current_epoch,
        best_val_loss,
        model,
        optimizer,
        scaler,
        scheduler,
        accumulation_steps,
        warmup_steps,
        total_steps,
    )


def train(
    rank: int,
    epoch: int,
    train_batch_size: int,
    train_dataloader: DataLoader,
    len_train_data: int,
    scaler: GradScaler,
    model: torch.nn.Module,
    tokenizer: whisper.tokenizer.Tokenizer,
    normalizer: EnglishTextNormalizer,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    accumulation_steps: int,
    max_grad_norm: float,
    train_res: Optional[wandb.Artifact],
    train_res_added: Optional[bool],
    tags: Optional[List[str]],
    exp_name: str,
) -> Tuple[
    torch.nn.Module, torch.optim.Optimizer, GradScaler, LambdaLR, Optional[bool]
]:
    """Training loop for 1 epoch

    Args:
        rank: The rank of the current process
        epoch: The current epoch number
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
        train_res: The training results artifact
        train_res_added: A boolean indicating whether the training results artifact has been added
        tags: The tags to use for logging
        exp_name: The experiment name

    Returns:
        A tuple containing the model, the optimizer, the gradient scaler, the scheduler, and a boolean indicating whether the training results artifact has been added
    """
    os.makedirs(f"logs/training/{exp_name}", exist_ok=True)
    batch_pred_text = []
    batch_tgt_text = []
    batch_unnorm_pred_text = []
    batch_audio_files = []
    batch_audio_arr = []
    batch_text_files = []
    if len_train_data % train_batch_size == 0:
        len_train_loader = len_train_data // train_batch_size
    else:
        len_train_loader = (len_train_data // train_batch_size) + 1
    logging_steps = (train_batch_size * accumulation_steps) // WANDB_EXAMPLES
    total_loss = 0.0
    model.train()
    optimizer.zero_grad()

    if rank == 0:
        train_table = wandb.Table(columns=for_logging.TRAIN_TABLE_COLS)
        start_time = time.time()

    for batch_idx, batch in enumerate(train_dataloader):
        if rank == 0:
            start_step = time.time()

        with autocast():
            (
                audio_files,
                transcript_files,
                audio_arr,
                audio_input,
                text_input,
                text_y,
                padding_mask,
            ) = batch

            # for logging purposes
            batch_audio_files.extend(audio_files)
            batch_audio_arr.extend(audio_arr)
            batch_text_files.extend(transcript_files)

            audio_input = audio_input.to(rank)
            text_input = text_input.to(rank)
            text_y = text_y.to(rank)
            padding_mask = padding_mask.to(rank)

            logits = model(audio_input, text_input, padding_mask)

            train_loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                text_y.view(-1),
                ignore_index=51864,
            )
            train_loss = (
                train_loss / accumulation_steps
            )  # normalization of loss (gradient accumulation)

        scaler.scale(train_loss).backward()  # accumulate gradients
        train_loss.detach_()
        total_loss += train_loss

        # alerting if loss is nan
        if rank == 0:
            if torch.isnan(train_loss):
                text = f"Loss is NaN for {audio_files} at epoch {epoch} and batch {batch_idx}!"
                wandb.alert(title="NaN Loss", text=text)

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
        batch_pred_text.extend(microbatch_pred_text)
        batch_unnorm_pred_text.extend(microbatch_unnorm_pred_text)

        microbatch_tgt_text = []
        for text_y_instance in text_y.cpu().numpy():
            tgt_y_instance_text = tokenizer.decode(list(text_y_instance))
            tgt_y_instance_text = tgt_y_instance_text.split("<|endoftext|>")[0]
            tgt_y_instance_text = tgt_y_instance_text + "<|endoftext|>"
            microbatch_tgt_text.append(tgt_y_instance_text)
        batch_tgt_text.extend(microbatch_tgt_text)

        # after accumulation_steps, update weights
        if ((batch_idx + 1) % accumulation_steps) == 0:
            train_loss_tensor = total_loss.clone()
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            train_loss_all = train_loss_tensor.item() / dist.get_world_size()

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
            else:
                train_wer = (
                    jiwer.wer(
                        reference=batch_tgt_text_full,
                        hypothesis=batch_pred_text_full,
                    )
                    * 100
                )
            # Use torch.tensor to work with dist.all_reduce
            train_wer_tensor = torch.tensor(train_wer, device=rank)
            # Aggregate WER across all processes
            dist.all_reduce(train_wer_tensor, op=dist.ReduceOp.SUM)
            # Calculate the average WER across all processes
            train_wer_all = train_wer_tensor.item() / dist.get_world_size()

            if rank == 0:
                print(f"{epoch=}")
                print(f"step={batch_idx + 1}")
                print(f"effective step={(batch_idx + 1) // accumulation_steps}")
                print(f"train_loss: {train_loss_all}")
                print(f"train_wer: {train_wer_all}")

                wandb.log(
                    {
                        "train/train_loss": train_loss_all,
                        "train/train_wer": train_wer_all,
                    }
                )

                if (
                    (batch_idx + 1)
                    % (
                        int(np.ceil((len_train_loader / accumulation_steps) / 10))
                        * accumulation_steps
                    )
                ) == 0:
                    with open(
                        f"logs/training/{exp_name}/training_results_{'_'.join(tags)}.txt",
                        "a",
                    ) as f:
                        if not train_res_added:  # only once
                            train_res.add_file(
                                f"logs/training/{exp_name}/training_results_{'_'.join(tags)}.txt"
                            )
                            train_res_added = True
                            wandb.log_artifact(train_res)

                        for i, (
                            tgt_text_instance,
                            pred_text_instance,
                        ) in enumerate(
                            norm_tgt_pred_pairs[
                                ::logging_steps
                            ]  # should log just 8 examples
                        ):
                            f.write(f"{epoch=}\n")
                            f.write(
                                f"effective step={(batch_idx + 1) // accumulation_steps}\n"
                            )
                            f.write(
                                f"text_file={batch_text_files[i * logging_steps]}\n"
                            )
                            f.write(f"{pred_text_instance=}\n")
                            f.write(
                                f"unnorm_pred_text_instance={batch_pred_text[i * logging_steps]}\n"
                            )
                            f.write(f"{tgt_text_instance=}\n")
                            f.write(
                                f"unnorm_tgt_text_instance={batch_tgt_text[i * logging_steps]}\n\n"
                            )

                        f.write(f"{train_loss_all=}\n")
                        f.write(f"{train_wer_all=}\n\n")

                if (batch_idx + 1) == (
                    int((len_train_loader / accumulation_steps) / 2)
                    * accumulation_steps
                ):
                    for i, (
                        tgt_text_instance,
                        pred_text_instance,
                    ) in enumerate(norm_tgt_pred_pairs[::logging_steps]):
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

                        train_table.add_data(
                            batch_audio_files[i * logging_steps],
                            wandb.Audio(
                                batch_audio_arr[i * logging_steps],
                                sample_rate=16000,
                            ),
                            batch_text_files[i * logging_steps],
                            pred_text_instance,
                            batch_unnorm_pred_text[i * logging_steps],
                            batch_pred_text[i * logging_steps],
                            tgt_text_instance,
                            batch_tgt_text[i * logging_steps],
                            subs,
                            dels,
                            ins,
                            len(tgt_text_instance.split()),
                            wer,
                        )

                    wandb.log({f"train_table_{epoch}": train_table})

            # Gradient clipping, if necessary, should be done before optimizer.step()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)  # Only update weights after accumulation_steps
            scaler.update()
            scheduler.step()  # Adjust learning rate based on accumulated steps
            current_lr = optimizer.param_groups[0]["lr"]
            # logging learning rate
            if rank == 0:
                wandb.log({"train/learning_rate": current_lr})
            optimizer.zero_grad()  # Reset gradients only after updating weights
            total_loss = 0.0

            batch_pred_text = []
            batch_tgt_text = []
            batch_unnorm_pred_text = []
            batch_audio_files = []
            batch_audio_arr = []
            batch_text_files = []

            if rank == 0:
                end_step = time.time()
                print(f"step {batch_idx + 1} took {(end_step - start_step) / 60.0} minutes")
                throughput = (train_batch_size * accumulation_steps / (end_step - start_step)) * 30 / 60
                print(f"audio_min_per_GPU_second: {throughput}")
                wandb.log({"train/time_step": (end_step - start_step) / 60.0})
                wandb.log({"train/audio_min_per_GPU_second": throughput})

    # If your dataset size is not a multiple of (batch_size * accumulation_steps)
    # Make sure to account for the last set of batches smaller than accumulation_steps
    if len_train_loader % accumulation_steps != 0:
        train_loss_tensor = total_loss.clone()
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss_all = train_loss_tensor.item() / dist.get_world_size()

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
        else:
            train_wer = (
                jiwer.wer(
                    reference=batch_tgt_text_full, hypothesis=batch_pred_text_full
                )
                * 100
            )

        # Use torch.tensor to work with dist.all_reduce
        train_wer_tensor = torch.tensor(train_wer, device=rank)
        # Aggregate WER across all processes
        dist.all_reduce(train_wer_tensor, op=dist.ReduceOp.SUM)
        # Calculate the average WER across all processes
        train_wer_all = train_wer_tensor.item() / dist.get_world_size()

        if rank == 0:
            print(f"last batch")
            print(f"{epoch=}")
            print(f"step={batch_idx + 1}")
            print(f"effective step={((batch_idx + 1) // accumulation_steps) + 1}")
            print(f"train_loss: {train_loss_all}")
            print(f"train_wer: {train_wer_all}")

            wandb.log(
                {"train/train_loss": train_loss_all, "train/train_wer": train_wer_all}
            )

            with open(
                f"logs/training/{exp_name}/training_results_{'_'.join(tags)}.txt",
                "a",
            ) as f:
                for i, (
                    tgt_text_instance,
                    pred_text_instance,
                ) in enumerate(norm_tgt_pred_pairs[::logging_steps]):
                    f.write(f"{epoch=}\n")
                    f.write(
                        f"effective step={((batch_idx + 1) // accumulation_steps) + 1}\n"
                    )
                    f.write(f"{batch_text_files[i * logging_steps]}\n")
                    f.write(f"{pred_text_instance=}\n")
                    f.write(
                        f"unnorm_pred_text_instance={batch_pred_text[i * logging_steps]}\n"
                    )
                    f.write(f"{tgt_text_instance=}\n")
                    f.write(
                        f"unnorm_tgt_text_instance={batch_tgt_text[i * logging_steps]}\n\n"
                    )

                f.write(f"{train_loss_all=}\n")
                f.write(f"{train_wer_all=}\n\n")

        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        if rank == 0:
            wandb.log({"train/learning_rate": current_lr})
        optimizer.zero_grad()
        total_loss = 0.0

        batch_pred_text = []
        batch_tgt_text = []
        batch_unnorm_pred_text = []
        batch_audio_files = []
        batch_audio_arr = []
        batch_text_files = []

    if rank == 0:
        end_time = time.time()
        with open(
            f"logs/training/{exp_name}/epoch_times_{'_'.join(tags)}.txt", "a"
        ) as f:
            f.write(
                f"train epoch {epoch} took {(end_time - start_time) / 60.0} minutes at effective step {(batch_idx + 1) // accumulation_steps}\n"
            )
            wandb.log({"train/time_epoch": (end_time - start_time) / 60.0})

    return model, optimizer, scaler, scheduler, train_res_added


def validate(
    rank: int,
    epoch: int,
    best_val_loss: Optional[float],
    len_val_data: int,
    val_batch_size: int,
    val_dataloader: DataLoader,
    scaler: GradScaler,
    model: torch.nn.Module,
    tokenizer: whisper.tokenizer.Tokenizer,
    normalizer: EnglishTextNormalizer,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    model_dims: ModelDimensions,
    model_variant: str,
    val_res: Optional[wandb.Artifact],
    val_res_added: Optional[bool],
    tags: Optional[List[str]],
    exp_name: str,
    run_id: str,
) -> Tuple[float, bool]:
    """Validation loop for 1 epoch

    Args:
        rank: The rank of the current process
        epoch: The current epoch number
        best_val_loss: The best validation loss
        val_dataloader: The dataloader for validation
        scaler: The gradient scaler
        model: The model to validate
        tokenizer: The tokenizer for encoding the text data
        normalizer: The text normalizer
        optimizer: The optimizer for training
        scheduler: The scheduler for training
        model_dims: The model dimensions
        model_variant: The variant of the model
        val_res: The validation results artifact
        val_res_added: A boolean indicating whether the validation results artifact has been added
        tags: The tags to use for logging
        exp_name: The experiment name
        run_id: The run ID

    Returns:
        A tuple containing the best validation loss and a boolean indicating whether the validation results artifact has been added
    """
    os.makedirs(f"logs/training/{exp_name}", exist_ok=True)
    if len_val_data % val_batch_size == 0:
        len_val_loader = len_val_data // val_batch_size
    else:
        len_val_loader = (len_val_data // val_batch_size) + 1
    val_loss = 0.0
    norm_pred_text = []
    norm_tgt_text = []

    if rank == 0:
        val_table = wandb.Table(columns=for_logging.VAL_TABLE_COLS)
        start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            with autocast():
                model.eval()
                (
                    audio_files,
                    transcript_files,
                    audio_arr,
                    audio_input,
                    text_input,
                    text_y,
                    padding_mask,
                ) = batch

                audio_input = audio_input.to(rank)
                text_input = text_input.to(rank)
                text_y = text_y.to(rank)
                padding_mask = padding_mask.to(rank)

                logits = model(audio_input, text_input, padding_mask)

                batch_val_loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    text_y.view(-1),
                    ignore_index=51864,
                )

                val_loss += batch_val_loss

            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)

            batch_pred_text = []
            unnorm_pred_text = []
            for pred_instance in pred.cpu().numpy():
                pred_instance_text = tokenizer.decode(list(pred_instance))
                unnorm_pred_text.append(pred_instance_text)
                pred_instance_text = ow.utils.remove_after_endoftext(pred_instance_text)
                batch_pred_text.append(pred_instance_text)

            batch_tgt_text = []
            for text_y_instance in text_y.cpu().numpy():
                tgt_y_instance_text = tokenizer.decode(list(text_y_instance))
                tgt_y_instance_text = tgt_y_instance_text.split("<|endoftext|>")[0]
                tgt_y_instance_text = tgt_y_instance_text + "<|endoftext|>"
                batch_tgt_text.append(tgt_y_instance_text)

            norm_batch_tgt_text = [normalizer(text) for text in batch_tgt_text]
            norm_batch_pred_text = [normalizer(text) for text in batch_pred_text]
            norm_tgt_pred_pairs = list(zip(norm_batch_tgt_text, norm_batch_pred_text))

            # no empty references - for WER calculation
            batch_tgt_text_full = [
                norm_batch_tgt_text[i]
                for i in range(len(norm_batch_tgt_text))
                if len(norm_batch_tgt_text[i]) > 0
            ]
            norm_tgt_text.extend(batch_tgt_text_full)
            batch_pred_text_full = [
                norm_batch_pred_text[i]
                for i in range(len(norm_batch_pred_text))
                if len(norm_batch_tgt_text[i]) > 0
            ]
            norm_pred_text.extend(batch_pred_text_full)

            if len(batch_tgt_text_full) == 0 and len(batch_pred_text_full) == 0:
                batch_val_wer = 0.0
            else:
                batch_val_wer = (
                    jiwer.wer(
                        reference=batch_tgt_text_full,
                        hypothesis=batch_pred_text_full,
                    )
                    * 100
                )

            if rank == 0:
                print(f"{epoch=}")
                print(f"val step={batch_idx + 1}")
                print(f"val_loss by batch: {batch_val_loss}")
                print(f"val_wer by batch: {batch_val_wer}")

                if (batch_idx + 1) % int(np.ceil(len_val_loader / 10)) == 0:
                    with open(
                        f"logs/training/{exp_name}/val_results_{'_'.join(tags)}.txt",
                        "a",
                    ) as f:
                        for i, (tgt_text_instance, pred_text_instance) in enumerate(
                            norm_tgt_pred_pairs
                        ):
                            if not val_res_added:  # only once
                                val_res.add_file(
                                    f"logs/training/{exp_name}/val_results_{'_'.join(tags)}.txt"
                                )
                                val_res_added = True
                                wandb.log_artifact(val_res)

                            f.write(f"{epoch=}\n")
                            f.write(f"{transcript_files[i]}\n")
                            f.write(f"{pred_text_instance=}\n")
                            f.write(
                                f"unnorm_pred_text_instance={batch_pred_text[i]=}\n"
                            )
                            f.write(f"{tgt_text_instance=}\n")
                            f.write(
                                f"unnorm_tgt_text_instance={batch_tgt_text[i]=}\n\n"
                            )

                        f.write(f"{batch_val_loss=}\n")
                        f.write(f"{batch_val_wer=}\n\n")

                if (batch_idx + 1) == (len_val_loader // 2):
                    for i, (tgt_text_instance, pred_text_instance) in enumerate(
                        norm_tgt_pred_pairs
                    ):
                        wer = np.round(
                            ow.utils.calculate_wer(
                                (tgt_text_instance, pred_text_instance)
                            ),
                            2,
                        )
                        subs = 0
                        dels = 0
                        ins = 0
                        if tgt_text_instance == "":
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
                            audio_files[i],
                            wandb.Audio(audio_arr[i], sample_rate=16000),
                            transcript_files[i],
                            pred_text_instance,
                            unnorm_pred_text[i],
                            batch_pred_text[i],
                            tgt_text_instance,
                            batch_tgt_text[i],
                            subs,
                            dels,
                            ins,
                            len(tgt_text_instance.split()),
                            wer,
                        )

                    wandb.log({f"val_table_{epoch}": val_table})

        if rank == 0:
            end_time = time.time()
            with open(
                f"logs/training/{exp_name}/epoch_times_{'_'.join(tags)}.txt", "a"
            ) as f:
                f.write(
                    f"val epoch {epoch} took {(end_time - start_time) / 60.0} minutes\n"
                )
                wandb.log({"val/time_epoch": (end_time - start_time) / 60.0})

        if len(norm_tgt_text) == 0 and len(norm_pred_text) == 0:
            val_wer = 0.0
        else:
            val_wer = (
                jiwer.wer(reference=norm_tgt_text, hypothesis=norm_pred_text) * 100
            )
            ave_val_loss = val_loss / len_val_loader

        val_wer_tensor = torch.tensor(val_wer, device=rank)
        dist.all_reduce(val_wer_tensor, op=dist.ReduceOp.SUM)
        val_wer_all = val_wer_tensor.item() / dist.get_world_size()

        ave_val_loss_tensor = ave_val_loss.clone()
        dist.all_reduce(ave_val_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss_all = ave_val_loss_tensor.item() / dist.get_world_size()

        if rank == 0:
            print(f"val_loss: {val_loss_all}")
            print(f"val_wer: {val_wer_all}")

            wandb.log({"val/val_loss": val_loss_all, "val/val_wer": val_wer_all})

            if val_loss_all < best_val_loss:
                best_val_loss = val_loss_all

                save_ckpt(
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    model_dims=model_dims,
                    tags=tags,
                    model_variant=model_variant,
                    exp_name=exp_name,
                    run_id=run_id,
                    file_name="best_val",
                )

    return best_val_loss, val_res_added


def evaluate(
    rank: int,
    epoch: int,
    eval_batch_size: int,
    num_workers: int,
    model: torch.nn.Module,
    normalizer: EnglishTextNormalizer,
    tags: List[str],
    exp_name: str,
) -> None:
    """Evaluation loop for 1 epoch

    Evaluation loop with WER calculation for 2 corpora: librispeech-clean and librispeech-other

    Args:
        rank: The rank of the current process
        epoch: The current epoch number
        eval_batch_size: The batch size for evaluation
        num_workers: The number of workers for the dataloader
        model: The model to evaluate
        normalizer: The text normalizer
        tags: The tags to use for logging
        exp_name: The experiment name
    """
    os.makedirs(f"logs/training/{exp_name}", exist_ok=True)
    eval_sets = ["librispeech_clean", "librispeech_other"]
    eval_table = wandb.Table(columns=for_logging.EVAL_TABLE_COLS)
    start_time = time.time()

    non_ddp_model = model.module
    non_ddp_model.eval()

    for eval_set in eval_sets:
        print(f"Evaluating {eval_set}\n")
        eval_dataset = EvalDataset(eval_set=eval_set)

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            persistent_workers=True,
        )

        hypotheses = []
        references = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(eval_dataloader), total=len(eval_dataloader)
            ):
                audio_input, text_y = batch
                audio_input = audio_input.to(rank)

                options = DecodingOptions(language="en", without_timestamps=True)

                results = non_ddp_model.decode(audio_input, options=options)
                norm_pred_text = [normalizer(result.text) for result in results]
                hypotheses.extend(norm_pred_text)
                norm_tgt_text = [normalizer(text) for text in text_y]
                references.extend(norm_tgt_text)

                if (batch_idx + 1) % int(np.ceil(len(eval_dataloader) / 10)) == 0:
                    for i in range(0, len(pred_text), 8):
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
                        eval_table.add_data(
                            eval_set,
                            audio_files[i],
                            wandb.Audio(audio_files[i], sample_rate=16000),
                            pred_text[i],
                            norm_pred_text[i],
                            norm_tgt_text[i],
                            wer,
                        )

                    wer = (
                        jiwer.wer(reference=norm_tgt_text, hypothesis=norm_pred_text)
                        * 100
                    )
                    with open(
                        f"logs/training/{exp_name}/eval_results_{'_'.join(tags)}.txt",
                        "a",
                    ) as f:
                        f.write(f"{eval_set} batch {batch_idx} WER: {wer}\n")

            avg_wer = jiwer.wer(references, hypotheses) * 100
            with open(
                f"logs/training/{exp_name}/eval_results_{'_'.join(tags)}.txt", "a"
            ) as f:
                f.write(f"{eval_set} average WER: {avg_wer}\n")
            wandb.log({f"eval/{eval_set}_wer": avg_wer})

    wandb.log({f"eval_table_{epoch}": eval_table})

    end_time = time.time()
    with open(f"logs/training/{exp_name}/epoch_times_{'_'.join(tags)}.txt", "a") as f:
        f.write(f"eval epoch {epoch} took {(end_time - start_time) / 60.0} minutes\n")
        wandb.log({"eval/time_epoch": (end_time - start_time) / 60.0})


def cleanup():
    """Cleanup function for the distributed training"""
    torch.cuda.empty_cache()
    dist.destroy_process_group()


def main(
    model_variant: str,
    exp_name: str,
    job_type: str,
    train_shards: str,
    val_shards: str,
    len_train_data: Optional[int],
    len_val_data: Optional[int],
    run_id: Optional[str] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    lr: float = 1.5e-3,
    betas: tuple = (0.9, 0.98),
    eps: float = 1e-6,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    epochs: int = 10,
    eff_size: int = 256,
    train_batch_size: int = 8,
    val_batch_size: int = 8,
    eval_batch_size: int = 32,
    num_workers: int = 10,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    run_eval: bool = False,
) -> None:
    """Main function for training

    Conducts a training loop for the specified number of epochs, with validation and evaluation (if run_eval is True)

    Args:
        model_variant: The variant of the model to use
        exp_name: The name of the experiment
        job_type: The type of job (e.g., training, evaluation)
        run_id: The run ID to use for loading a checkpoint
        rank: The rank of the current process
        world_size: The total number of processes
        lr: The learning rate
        betas: The betas for the optimizer
        eps: The epsilon for the optimizer
        weight_decay: The weight decay for the optimizer
        max_grad_norm: The maximum gradient norm
        epochs: The number of epochs to train for
        eff_size: The size of the efficientnet model
        train_batch_size: The batch size for training
        val_batch_size: The batch size for validation
        eval_batch_size: The batch size for evaluation
        num_workers: The number of workers for the dataloader
        pin_memory: Whether to pin memory for the dataloader
        persistent_workers: Whether to use persistent workers for the dataloader
        run_eval: Whether to run evaluation
    """
    model_dims = VARIANT_TO_DIMS[model_variant]

    if rank is None and world_size is None:
        rank = int(os.getenv("RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))

    # setup the process groups
    setup(rank, world_size)

    # setup the tokenizer and normalizer
    tokenizer = get_tokenizer(multilingual=False)
    normalizer = EnglishTextNormalizer()
    n_text_ctx = model_dims.n_text_ctx

    # prepare dataset
    train_dataloader, len_train_data, val_dataloader, len_val_data = prepare_data(
        rank=rank,
        world_size=world_size,
        train_shards=train_shards,
        val_shards=val_shards,
        len_train_data=len_train_data,
        len_val_data=len_val_data,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        n_text_ctx=n_text_ctx,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    # model instantiation
    if run_id is not None:
        (
            current_epoch,
            best_val_loss,
            model,
            optimizer,
            scaler,
            scheduler,
            accumulation_steps,
            warmup_steps,
            total_steps,
        ) = load_ckpt(
            exp_name=exp_name,
            run_id=run_id,
            rank=rank,
            world_size=world_size,
            epochs=epochs,
            train_batch_size=train_batch_size,
            eff_size=eff_size,
            len_train_data=len_train_data,
        )
    else:
        model = ow.model.Whisper(dims=model_dims).to(rank)
        model = DDP(model, device_ids=[rank], output_device=rank)

        # optimizer and scheduler instantiation
        optimizer = prepare_optim(
            model=model, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

        scheduler, accumulation_steps, warmup_steps, total_steps = prepare_sched(
            len_train_data=len_train_data,
            world_size=world_size,
            train_batch_size=train_batch_size,
            eff_size=eff_size,
            epochs=epochs,
            optimizer=optimizer,
        )

        scaler = GradScaler()

        current_epoch = 0

    # setting up wandb for logging
    if rank == 0:
        if run_id is None:
            best_val_loss = float("inf")

        run_id, tags, train_res, val_res, train_res_added, val_res_added = setup_wandb(
            run_id=run_id,
            exp_name=exp_name,
            job_type=job_type,
            model_variant=model_variant,
            lr=lr,
            betas=betas,
            eps=eps,
            len_train_data=len_train_data,
            len_val_data=len_val_data,
            weight_decay=weight_decay,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            epochs=epochs,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            accumulation_steps=accumulation_steps,
            world_size=world_size,
            num_workers=num_workers,
        )

    for epoch in range(current_epoch, epochs):
        model, optimizer, scaler, scheduler, train_res_added = train(
            rank=rank,
            epoch=epoch,
            train_batch_size=train_batch_size,
            train_dataloader=train_dataloader,
            len_train_data=len_train_data,
            scaler=scaler,
            model=model,
            tokenizer=tokenizer,
            normalizer=normalizer,
            optimizer=optimizer,
            scheduler=scheduler,
            accumulation_steps=accumulation_steps,
            max_grad_norm=max_grad_norm,
            train_res=train_res if rank == 0 else None,
            train_res_added=train_res_added if rank == 0 else None,
            tags=tags if rank == 0 else None,
            exp_name=exp_name,
        )

        if rank == 0:
            save_ckpt(
                epoch=epoch,
                best_val_loss=best_val_loss,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                model_dims=model_dims,
                tags=tags,
                model_variant=model_variant,
                exp_name=exp_name,
                run_id=run_id,
                file_name="latest_train",
            )

        best_val_loss, val_res_added = validate(
            rank=rank,
            epoch=epoch,
            best_val_loss=best_val_loss if rank == 0 else None,
            len_val_data=len_val_data,
            val_batch_size=val_batch_size,
            val_dataloader=val_dataloader,
            scaler=scaler,
            model=model,
            tokenizer=tokenizer,
            normalizer=normalizer,
            optimizer=optimizer,
            scheduler=scheduler,
            model_dims=model_dims,
            model_variant=model_variant,
            val_res=val_res if rank == 0 else None,
            val_res_added=val_res_added if rank == 0 else None,
            tags=tags if rank == 0 else None,
            exp_name=exp_name,
            run_id=run_id,
        )

        if run_eval:
            if rank != 0:
                dist.barrier()

            if rank == 0:
                evaluate(
                    rank=rank,
                    epoch=epoch,
                    eval_batch_size=eval_batch_size,
                    num_workers=num_workers,
                    model=model,
                    normalizer=normalizer,
                    tags=tags,
                    exp_name=exp_name,
                )

            if rank == 0:
                dist.barrier()

    cleanup()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    Fire(main)
