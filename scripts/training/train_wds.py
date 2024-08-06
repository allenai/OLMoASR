import json
import os
import glob
from io import BytesIO
import numpy as np
import wandb
from typing import List, Tuple, Union, Optional, Literal, Dict
import time
from datetime import timedelta
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
import webdataset as wds

import whisper
from whisper import audio, DecodingOptions
from whisper.normalizers import EnglishTextNormalizer
from whisper.tokenizer import get_tokenizer
import whisper.tokenizer
from open_whisper.config.model_dims import VARIANT_TO_DIMS, ModelDimensions
import open_whisper as ow

from scripts.eval.eval import EvalDataset
from scripts.training import for_logging

WANDB_EXAMPLES = 8
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TORCH_DISTRIBUTED_DETAIL"] = "DEBUG"


def decode_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """Decodes audio bytes to numpy array

    Args:
        audio_bytes: The audio bytes to decode

    Returns:
        The audio array
    """
    bytes_io = BytesIO(audio_bytes)
    audio_arr = np.load(bytes_io)

    return audio_arr


def decode_text_bytes(text_bytes):
    """
    Decode the given bytes object into a string using the UTF-8 encoding.

    Args:
        text_bytes: The bytes object to decode.

    Returns:
        The decoded string.
    """
    transcript_str = text_bytes.decode("utf-8")

    return transcript_str


def decode_sample(sample: Dict[str, bytes]) -> Tuple[str, np.ndarray, str, str]:
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
    file_path = os.path.join(sample["__url__"], sample["__key__"])
    audio_path = file_path + ".npy"
    text_path = file_path + ".srt"
    audio_bytes = sample["npy"]
    text_bytes = sample["srt"]
    audio_arr = decode_audio_bytes(audio_bytes)
    transcript_str = decode_text_bytes(text_bytes)

    return audio_path, audio_arr, text_path, transcript_str


def preprocess_audio(audio_arr: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocesses the given audio array.

    Args:
        audio_arr: The input audio array.

    Returns:
        A tuple containing the preprocessed mel spectrogram and the original audio array.
    """
    audio_arr = audio_arr.astype(np.float32) / 32768.0
    audio_arr = audio.pad_or_trim(audio_arr)
    mel_spec = audio.log_mel_spectrogram(audio_arr)

    return mel_spec, audio_arr


def preprocess_text(
    transcript_string: str, tokenizer: whisper.tokenizer.Tokenizer, n_text_ctx: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preprocesses the transcript text by tokenizing it, adding special tokens,
    padding it to a fixed length, and creating a padding mask.

    Args:
        transcript_string: The input transcript string.
        tokenizer: The tokenizer object used for tokenization.
        n_text_ctx: The length of the text context.

    Returns:
        A tuple containing the preprocessed text input, text output, and padding mask.
    """
    reader = ow.utils.TranscriptReader(transcript_string=transcript_string, ext="srt")
    transcript, *_ = reader.read()

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


def preprocess(sample, tokenizer, n_text_ctx):
    """
    Preprocesses the given sample by performing audio and text preprocessing.

    Args:
        sample: A tuple containing audio and text information.
            - audio_path: Path to the audio file.
            - audio_arr: Array containing the audio data.
            - text_path: Path to the text file.
            - transcript_str: Transcription of the audio.

        tokenizer: The tokenizer object used for text preprocessing.
        n_text_ctx: The number of context tokens to consider for text preprocessing.

    Returns:
        A tuple containing the preprocessed data.
            - audio_path: Path to the audio file.
            - text_path: Path to the text file.
            - padded_audio_arr: Padded array containing the preprocessed audio data.
            - audio_input: Tensor containing the preprocessed audio input.
            - text_input: Tensor containing the preprocessed text input.
            - text_y: Tensor containing the preprocessed text output.
            - padding_mask: Tensor containing the padding mask for the text input.
    """
    audio_path, audio_arr, text_path, transcript_str = sample
    audio_input, padded_audio_arr = preprocess_audio(audio_arr)
    text_input, text_y, padding_mask = preprocess_text(
        transcript_str, tokenizer, n_text_ctx
    )

    return (
        audio_path,
        text_path,
        padded_audio_arr,
        audio_input,
        text_input,
        text_y,
        padding_mask,
    )


def wds_pipeline(
    shards,
    batch_size,
    n_text_ctx,
    tokenizer,
    val_flag,
):
    """
    Creates a data pipeline using the webdataset library.

    Args:
        shards: The path to the shards.
        batch_size: The batch size for the data pipeline.
        n_text_ctx: The number of text contexts.
        tokenizer: The tokenizer object.

    Returns:
        wds.DataPipeline: The data pipeline object.
    """
    if val_flag is False:
        dataset = wds.DataPipeline(
            wds.SimpleShardList(shards),
            wds.split_by_node,
            wds.split_by_worker,
            wds.shuffle(bufsize=1000, initial=100),
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(),
            # this shuffles the samples in memory
            wds.shuffle(bufsize=1000, initial=100),
            wds.map(decode_sample),
            wds.map(lambda sample: preprocess(sample, tokenizer, n_text_ctx)),
            wds.shuffle(bufsize=1000, initial=100),
            wds.batched(batch_size),
        )
    else:
        dataset = wds.DataPipeline(
            wds.SimpleShardList(shards),
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(),
            wds.map(decode_sample),
            wds.map(lambda sample: preprocess(sample, tokenizer, n_text_ctx)),
            wds.batched(batch_size),
        )

    return dataset


def setup(rank: int, world_size: int) -> None:
    """Initializes the distributed process group

    Args:
        rank: The rank of the current process
        world_size: The total number of processes
    """
    dist.init_process_group(
        "nccl", timeout=timedelta(seconds=3600), rank=rank, world_size=world_size
    )


def prepare_dataloader(
    dataset: Dataset,
    train_samples: Optional[int],
    pin_memory: bool,
    num_workers: int,
    persistent_workers: bool,
    val_flag: bool,
    batch_size: Optional[int] = None,
) -> wds.WebLoader:
    """Prepares the dataloader for the dataset

    Prepares the distributed sampler and the dataloader for the dataset for DDP training

    Args:
        dataset: The dataset to use
        pin_memory: Whether to pin memory
        num_workers: The number of workers
        persistent_workers: Whether to use persistent workers
        val_flag: Whether the dataloader is for validation
        batch_size: The batch size

    Returns:
        WebDataset WebLoader
    """
    if val_flag is False:
        print(f"{os.environ['WORLD_SIZE']=}")
        epoch_steps = int(
            np.ceil(train_samples / (int(os.environ["WORLD_SIZE"]) * batch_size))
        )
        print(f"{epoch_steps=}")
        dataloader = (
            wds.WebLoader(
                dataset,
                batch_size=None,
                shuffle=False,
                pin_memory=pin_memory,
                num_workers=num_workers,
                drop_last=False,
                persistent_workers=persistent_workers,
            )
            .repeat(2)
            .with_epoch(epoch_steps)
        )
    else:
        dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=False,
            persistent_workers=persistent_workers,
        )

    return dataloader


def prepare_data(
    rank: Optional[int],
    train_shards: str,
    val_shards: Optional[str],
    train_batch_size: int,
    val_batch_size: int,
    train_samples: int,
    n_text_ctx: int,
    tokenizer: whisper.tokenizer.Tokenizer,
    pin_memory: bool = True,
    num_workers: int = 1,
    persistent_workers: bool = True,
) -> Tuple[wds.WebLoader, Optional[wds.WebLoader]]:
    """Prepares the data for training

    Args:
        rank: The rank of the current process
        train_shards: The path to the training shards
        val_shards: The path to the validation shards
        train_batch_size: The batch size for training
        val_batch_size: The batch size for validation
        train_samples: The number of training samples
        n_text_ctx: The number of text tokens
        tokenizer: The tokenizer for tokenizing the text data
        pin_memory: Whether to pin memory
        num_workers: The number of workers
        persistent_workers: Whether to use persistent workers

    Returns:
        A tuple containing dataloader for training and validation
    """
    train_dataset = wds_pipeline(
        shards=train_shards,
        batch_size=train_batch_size,
        n_text_ctx=n_text_ctx,
        tokenizer=tokenizer,
        val_flag=False,
    )

    train_dataloader = prepare_dataloader(
        dataset=train_dataset,
        train_samples=train_samples,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        val_flag=False,
        batch_size=train_batch_size,
    )

    if val_shards is not None:
        val_dataset = wds_pipeline(
            shards=val_shards,
            batch_size=val_batch_size,
            n_text_ctx=n_text_ctx,
            tokenizer=tokenizer,
            val_flag=True,
        )

        val_dataloader = prepare_dataloader(
            dataset=val_dataset,
            train_samples=None,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            val_flag=True,
        )

        return train_dataloader, val_dataloader

    return train_dataloader, None


def prepare_eval_data(
    eval_set: str, eval_batch_size: int, num_workers: int
) -> DataLoader:
    eval_dataset = EvalDataset(eval_set=eval_set)

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=True,
        pin_memory=True,
    )

    return eval_dataloader


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
        epochs: The number of epochs
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

    def lr_lambda(batch_idx: int) -> float:
        eff_batch_idx = batch_idx // accumulation_steps
        if eff_batch_idx < warmup_steps:
            return float(eff_batch_idx) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(train_steps - eff_batch_idx)
            / float(max(1, train_steps - warmup_steps)),
        )

    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

    return scheduler, accumulation_steps, warmup_steps, train_steps


def setup_wandb(
    run_id: Optional[str],
    exp_name: str,
    job_type: str,
    model_variant: str,
    train_shards: str,
    train_steps: int,
    warmup_steps: int,
    accumulation_steps: int,
    world_size: int,
    num_workers: int,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    eff_batch_size: int,
    train_batch_size: int,
    val_batch_size: int,
) -> Tuple[Optional[str], List[str], wandb.Artifact, wandb.Artifact, bool, bool]:
    """Sets up the Weights and Biases logging

    Args:
        run_id: The run ID
        exp_name: The experiment name
        job_type: The type of job
        model_variant: The variant of the model
        train_steps: The total number of steps
        warmup_steps: The number of warmup steps
        accumulation_steps: The number of steps over which to accumulate gradients
        world_size: The total number of processes
        num_workers: The number of workers
        lr: The learning rate
        betas: The betas for the Adam optimizer
        eps: The epsilon value
        weight_decay: The weight decay
        eff_batch_size: The effective train batch size
        train_batch_size: The batch size for training
        val_batch_size: The batch size for validation

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
        "val_batch_size": val_batch_size,
        "train_steps": train_steps,
        "warmup_steps": warmup_steps,
        "accumulation_steps": accumulation_steps,
        "world_size": world_size,
        "num_workers": num_workers,
        "model_variant": model_variant,
        "train_shards": train_shards.split("/")[-1],
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
    current_step: int,
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
    """
    ddp_checkpoint = {
        "current_step": current_step,
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
        "current_step": current_step,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        # You can also save other items such as scheduler state
        "scheduler_state_dict": (scheduler.state_dict() if scheduler else None),
        "dims": model_dims,
    }

    os.makedirs(f"checkpoints/{exp_name}_{run_id}", exist_ok=True)

    torch.save(
        ddp_checkpoint,
        f"checkpoints/{exp_name}_{run_id}/{file_name}_{model_variant}_{'_'.join(tags)}_ddp.pt",
    )
    torch.save(
        non_ddp_checkpoint,
        f"checkpoints/{exp_name}_{run_id}/{file_name}_{model_variant}_{'_'.join(tags)}_non_ddp.pt",
    )


def load_ckpt(
    exp_name: str,
    run_id: str,
    rank: int,
    world_size: int,
    train_steps: int,
    train_batch_size: int,
    eff_batch_size: int,
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
        train_steps: The total number of steps for training
        train_batch_size: The batch size for training
        eff_batch_size: The effective batch size for training

    Returns:
        A tuple containing the current step, the best validation loss, the model, the optimizer, the gradient scaler,
        the scheduler, the number of steps over which to accumulate gradients, the number of warmup steps, and the total number of steps
    """
    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}

    ckpt_file = glob.glob(
        f"checkpoints/{exp_name}_{run_id}/latest_train_*_fp16_ddp.pt"
    )[0]

    ckpt = torch.load(ckpt_file, map_location=map_location)

    model = ow.model.Whisper(dims=ckpt["dims"]).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # if end at training step i, then start at step i+1 when resuming
    current_step = ckpt["current_step"] + 1

    best_val_loss = ckpt["best_val_loss"]

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
        current_step,
        best_val_loss,
        model,
        optimizer,
        scaler,
        scheduler,
        accumulation_steps,
        warmup_steps,
        train_steps,
    )


def train(
    rank: int,
    current_step: int,
    train_batch_size: int,
    train_dataloader: wds.WebLoader,
    train_steps: int,
    scaler: GradScaler,
    model: DDP,
    tokenizer: whisper.tokenizer.Tokenizer,
    normalizer: EnglishTextNormalizer,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    accumulation_steps: int,
    max_grad_norm: float,
    train_res: Optional[wandb.Artifact],
    train_res_added: Optional[bool],
    run_val: bool,
    val_dataloader: Optional[wds.WebLoader],
    model_dims: Optional[ModelDimensions],
    model_variant: Optional[str],
    best_val_loss: Optional[float],
    val_res: Optional[wandb.Artifact],
    val_res_added: Optional[bool],
    run_eval: bool,
    eval_batch_size: int,
    eval_num_workers: int,
    eval_loaders: List[DataLoader],
    run_id: Optional[str],
    tags: Optional[List[str]],
    exp_name: Optional[str],
) -> Tuple[
    int,
    Optional[float],
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
        train_steps: The total number of steps for training
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
        A tuple containing the current step, model, the optimizer, the gradient scaler, the scheduler, and a boolean indicating whether the training results artifact has been added
    """
    batch_pred_text = []
    batch_tgt_text = []
    batch_unnorm_pred_text = []
    batch_audio_files = []
    batch_text_files = []
    batch_audio_arr = []
    logging_steps = (train_batch_size * accumulation_steps) // WANDB_EXAMPLES
    total_loss = 0.0
    model.train()
    optimizer.zero_grad()

    if rank == 0:
        train_table = wandb.Table(columns=for_logging.TRAIN_TABLE_COLS)
        start_time = time.time()
    
    for batch_idx, batch in enumerate(train_dataloader):
        dist.barrier()
        start_step = time.time()

        with autocast():
            (
                audio_files,
                transcript_files,
                padded_audio_arr,
                audio_input,
                text_input,
                text_y,
                padding_mask,
            ) = batch

            # for logging purposes
            batch_audio_files.extend(audio_files)
            batch_text_files.extend(transcript_files)
            batch_audio_arr.extend(padded_audio_arr)

            audio_input = audio_input.to(rank)
            text_input = text_input.to(rank)
            text_y = text_y.to(rank)
            padding_mask = padding_mask.to(rank)

            # forward pass
            logits = model(audio_input, text_input, padding_mask)

            # calculate loss
            train_loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                text_y.view(-1),
                ignore_index=51864,
            )

            train_loss = (
                train_loss / accumulation_steps
            )  # normalization of loss (gradient accumulation)
        # backpropagation
        scaler.scale(train_loss).backward()  # accumulate gradients
        train_loss.detach_()
        total_loss += train_loss

        # alerting if loss is nan
        if rank == 0:
            if torch.isnan(train_loss):
                text = f"Loss is NaN for {audio_files} at step {current_step}!"
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

        dist.barrier()
        # after accumulation_steps, update weights
        if ((batch_idx + 1) % accumulation_steps) == 0:
            print(f"{rank=}, {batch_idx=}")
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
            # dist.barrier()
            # Aggregate WER across all processes
            dist.all_reduce(train_wer_tensor, op=dist.ReduceOp.SUM)
            # Calculate the average WER across all processes
            train_wer_all = train_wer_tensor.item() / dist.get_world_size()

            # update weights
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)  # gradient clipping
            scaler.step(optimizer)  # Only update weights after accumulation_steps
            scaler.update()
            scheduler.step()  # Adjust learning rate based on accumulated steps

            current_step += 1
            if rank == 0:
                wandb.log({"train/step": current_step})

            if current_step >= train_steps:
                return (
                    current_step,
                    best_val_loss,
                    model,
                    optimizer,
                    scaler,
                    scheduler,
                    train_res_added,
                    val_res_added,
                )

            end_step = time.time()
            time_per_step = (end_step - start_step) / 60
            throughput = (
                ((train_batch_size * accumulation_steps) / (end_step - start_step))
                * 30
                / 60
            )

            # putting throughput on GPU
            throughput_tensor = torch.tensor(throughput, device=rank)
            time_tensor = torch.tensor(time_per_step, device=rank)
            # prepare list to gather throughput from all processes
            if rank == 0:
                gathered_throughput = [
                    torch.zeros_like(throughput_tensor).to(rank)
                    for _ in range(dist.get_world_size())
                ]
                gathered_time = [
                    torch.zeros_like(time_tensor).to(rank)
                    for _ in range(dist.get_world_size())
                ]
            else:
                gathered_throughput = None
                gathered_time = None

            dist.gather(throughput_tensor, gather_list=gathered_throughput, dst=0)
            dist.gather(time_tensor, gather_list=gathered_time, dst=0)

            if rank == 0:
                gathered_throughput = [t.item() for t in gathered_throughput]
                gathered_time = [t.item() for t in gathered_time]
                for i, throughput in enumerate(gathered_throughput):
                    wandb.log({f"train/audio_min_per_GPU_second_gpu={i}": throughput})

                for i, time_per_step in enumerate(gathered_time):
                    wandb.log({f"train/time_per_step_gpu={i}": time_per_step})

            current_lr = optimizer.param_groups[0]["lr"]
            # logging learning rate
            if rank == 0:
                wandb.log({"train/learning_rate": current_lr})
            optimizer.zero_grad()  # Reset gradients only after updating weights
            total_loss = 0.0
            
            # logging
            if rank == 0:
                print(f"current_step: {current_step}")
                print(f"train_loss: {train_loss_all}")
                print(f"train_wer: {train_wer_all}")

                wandb.log(
                    {
                        "train/train_loss": train_loss_all,
                        "train/train_wer": train_wer_all,
                    }
                )

                if (current_step % (int(np.ceil(train_steps / 10000)))) == 0:
                    os.makedirs(f"logs/training/{exp_name}/{run_id}", exist_ok=True)
                    with open(
                        f"logs/training/{exp_name}/{run_id}/training_results_{'_'.join(tags)}.txt",
                        "a",
                    ) as f:
                        if not train_res_added:  # only once
                            train_res.add_file(
                                f"logs/training/{exp_name}/{run_id}/training_results_{'_'.join(tags)}.txt"
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
                            f.write(f"{current_step=}\n")
                            f.write(
                                f"effective step in epoch={(batch_idx + 1) // accumulation_steps}\n"
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

                if (current_step % (int(np.ceil(train_steps / 200)))) == 0:
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

                    wandb.log({f"train_table_{current_step}": train_table})
                    train_table = wandb.Table(columns=for_logging.TRAIN_TABLE_COLS)


            # validation
            if run_val:
                if (
                    current_step % (int(np.ceil(train_steps / 200)))
                ) == 0 and current_step > 0:
                    best_val_loss, val_res_added = validate(
                        rank=rank,
                        current_step=current_step,
                        best_val_loss=best_val_loss,
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
                        exp_name=exp_name if rank == 0 else None,
                        run_id=run_id if rank == 0 else None,
                    )

                if (
                    current_step % (int(np.ceil(train_steps / 200)))
                ) == 0 and current_step > 0:
                    print(f"Rank {rank} reaching barrier w/ best val loss {best_val_loss}")
                dist.barrier()
                if (
                    current_step % (int(np.ceil(train_steps / 200)))
                ) == 0 and current_step > 0:
                    print(f"Rank {rank} passing barrier w/ best val loss {best_val_loss}")

            # evaluation
            if run_eval:
                if (
                    current_step % (int(np.ceil(train_steps / 20)))
                ) == 0 and current_step > 0:
                    evaluate(
                        rank=rank,
                        current_step=current_step,
                        eval_batch_size=eval_batch_size,
                        num_workers=eval_num_workers,
                        model=model,
                        eval_loaders=eval_loaders,
                        normalizer=normalizer,
                        tags=tags if rank == 0 else None,
                        exp_name=exp_name if rank == 0 else None,
                        run_id=run_id if rank == 0 else None,
                    )

                if (
                    current_step % (int(np.ceil(train_steps / 20)))
                ) == 0 and current_step > 0:
                    print(f"Rank {rank} reaching barrier")
                dist.barrier()
                if (
                    current_step % (int(np.ceil(train_steps / 20)))
                ) == 0 and current_step > 0:
                    print(f"Rank {rank} passing barrier")

            batch_pred_text = []
            batch_tgt_text = []
            batch_unnorm_pred_text = []
            batch_audio_files = []
            batch_text_files = []
            batch_audio_arr = []

    # If your dataset size is not a multiple of (batch_size * accumulation_steps)
    # Make sure to account for the last set of batches smaller than accumulation_steps
    # dist.barrier()

    if total_loss > 0.0:
        train_loss_tensor = total_loss.clone()
        # dist.barrier()
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
        # dist.barrier()
        # Aggregate WER across all processes
        dist.all_reduce(train_wer_tensor, op=dist.ReduceOp.SUM)
        # Calculate the average WER across all processes
        train_wer_all = train_wer_tensor.item() / dist.get_world_size()

        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        current_step += 1
        if current_step >= train_steps:
            return (
                current_step,
                best_val_loss,
                model,
                optimizer,
                scaler,
                scheduler,
                train_res_added,
                val_res_added,
            )

        current_lr = optimizer.param_groups[0]["lr"]
        if rank == 0:
            wandb.log({"train/learning_rate": current_lr})
        optimizer.zero_grad()
        total_loss = 0.0

        if rank == 0:
            print(f"current_step: {current_step}")
            print(f"train_loss: {train_loss_all}")
            print(f"train_wer: {train_wer_all}")

            wandb.log(
                {
                    "train/train_loss": train_loss_all,
                    "train/train_wer": train_wer_all,
                }
            )

            os.makedirs(f"logs/training/{exp_name}/{run_id}", exist_ok=True)
            with open(
                f"logs/training/{exp_name}/{run_id}/training_results_{'_'.join(tags)}.txt",
                "a",
            ) as f:
                for i, (
                    tgt_text_instance,
                    pred_text_instance,
                ) in enumerate(norm_tgt_pred_pairs[::logging_steps]):
                    f.write(f"{current_step=}\n")
                    f.write(
                        f"effective step in epoch={(batch_idx + 1) // accumulation_steps}\n"
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

        # dist.barrier()

        batch_pred_text = []
        batch_tgt_text = []
        batch_unnorm_pred_text = []
        batch_audio_files = []
        batch_text_files = []
        batch_audio_arr = []

    if rank == 0:
        end_time = time.time()
        os.makedirs(f"logs/training/{exp_name}/{run_id}", exist_ok=True)
        with open(
            f"logs/training/{exp_name}/{run_id}/epoch_times_{'_'.join(tags)}.txt", "a"
        ) as f:
            f.write(
                f"train epoch took {(end_time - start_time) / 60} minutes at effective step {(batch_idx + 1) // accumulation_steps}\n"
            )
            wandb.log({"train/time_epoch": (end_time - start_time) / 60})

    # dist.barrier()

    return (
        current_step,
        best_val_loss,
        model,
        optimizer,
        scaler,
        scheduler,
        train_res_added,
        val_res_added,
    )


def validate(
    rank: int,
    current_step: int,
    best_val_loss: float,
    val_dataloader: DataLoader,
    scaler: GradScaler,
    model: DDP,
    tokenizer: whisper.tokenizer.Tokenizer,
    normalizer: EnglishTextNormalizer,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    model_dims: ModelDimensions,
    model_variant: str,
    val_res: Optional[wandb.Artifact],
    val_res_added: Optional[bool],
    tags: Optional[List[str]],
    exp_name: Optional[str],
    run_id: Optional[str],
) -> Tuple[float, bool]:
    """Validation loop for 1 epoch

    Args:
        rank: The rank of the current process
        current_step: The current step
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
    val_loss = 0.0
    ave_val_loss = torch.tensor(0.0, device=rank)
    val_steps = 0
    norm_pred_text = []
    norm_tgt_text = []
    non_ddp_model = model.module

    if rank == 0:
        val_table = wandb.Table(columns=for_logging.VAL_TABLE_COLS)
        start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            with autocast():
                non_ddp_model.eval()
                (
                    audio_files,
                    transcript_files,
                    padded_audio_arr,
                    audio_input,
                    text_input,
                    text_y,
                    padding_mask,
                ) = batch

                audio_input = audio_input.to(rank)
                text_input = text_input.to(rank)
                text_y = text_y.to(rank)
                padding_mask = padding_mask.to(rank)

                logits = non_ddp_model(audio_input, text_input, padding_mask)

                batch_val_loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    text_y.view(-1),
                    ignore_index=51864,
                )

                val_loss += batch_val_loss
                val_steps += 1

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
                print(f"val step={batch_idx + 1}")
                print(f"val_loss by batch: {batch_val_loss}")
                print(f"val_wer by batch: {batch_val_wer}")

                if (batch_idx + 1) % 10 == 0:
                    with open(
                        f"logs/training/{exp_name}/{run_id}/val_results_{'_'.join(tags)}.txt",
                        "a",
                    ) as f:
                        for i, (tgt_text_instance, pred_text_instance) in enumerate(
                            norm_tgt_pred_pairs
                        ):
                            if not val_res_added:  # only once
                                val_res.add_file(
                                    f"logs/training/{exp_name}/{run_id}/val_results_{'_'.join(tags)}.txt"
                                )
                                val_res_added = True
                                wandb.log_artifact(val_res)

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

                if (batch_idx + 1) % 20 == 0:
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
                            wandb.Audio(padded_audio_arr[i], sample_rate=16000),
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

        if rank == 0:
            wandb.log({f"val_table_{current_step}": val_table})
            end_time = time.time()
            with open(
                f"logs/training/{exp_name}/{run_id}/epoch_times_{'_'.join(tags)}.txt",
                "a",
            ) as f:
                f.write(f"val epoch took {(end_time - start_time) / 60.0} minutes\n")
                wandb.log({"val/time_epoch": (end_time - start_time) / 60.0})

        if len(norm_tgt_text) == 0 and len(norm_pred_text) == 0:
            val_wer = 0.0 * 100
        else:
            val_wer = (
                jiwer.wer(reference=norm_tgt_text, hypothesis=norm_pred_text) * 100
            )
            ave_val_loss = val_loss / val_steps

        if rank == 0:
            print(f"best_val_loss: {best_val_loss}")
            print(f"val_loss: {ave_val_loss}")
            print(f"val_wer: {val_wer}")

            wandb.log({"val/val_loss": ave_val_loss, "val/val_wer": val_wer})

            if ave_val_loss < best_val_loss:
                best_val_loss = ave_val_loss
                print("Saving best model")
                save_ckpt(
                    current_step=current_step,
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
    current_step: int,
    eval_batch_size: int,
    num_workers: int,
    model: torch.nn.Module,
    eval_loaders: List[DataLoader],
    normalizer: EnglishTextNormalizer,
    tags: Optional[List[str]],
    exp_name: Optional[str],
    run_id: Optional[str],
) -> None:
    """Evaluation loop for 1 epoch

    Evaluation loop with WER calculation for 2 corpora: librispeech-clean and librispeech-other

    Args:
        rank: The rank of the current process
        current_step: The current step
        eval_batch_size: The batch size for evaluation
        num_workers: The number of workers for the dataloader
        model: The model to evaluate
        normalizer: The text normalizer
        tags: The tags to use for logging
        exp_name: The experiment name
    """
    eval_table = wandb.Table(columns=for_logging.EVAL_TABLE_COLS)
    start_time = time.time()

    non_ddp_model = model.module
    non_ddp_model.eval()

    for eval_set, eval_dataloader in eval_loaders:
        print(f"Evaluating {eval_set}\n")

        hypotheses = []
        references = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(eval_dataloader), total=len(eval_dataloader)
            ):
                audio_fp, _, audio_input, text_y = batch
                audio_input = audio_input.to(rank)

                options = DecodingOptions(language="en", without_timestamps=True)

                results = non_ddp_model.decode(audio_input, options=options)
                pred_text = [result.text for result in results]
                norm_pred_text = [normalizer(text) for text in pred_text]
                hypotheses.extend(norm_pred_text)
                norm_tgt_text = [normalizer(text) for text in text_y]
                references.extend(norm_tgt_text)

                if rank == 0:
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
                                wandb.Audio(audio_fp[i], sample_rate=16000),
                                pred_text[i],
                                norm_pred_text[i],
                                norm_tgt_text[i],
                                wer,
                            )

                        wer = (
                            jiwer.wer(
                                reference=norm_tgt_text, hypothesis=norm_pred_text
                            )
                            * 100
                        )
                        with open(
                            f"logs/training/{exp_name}/{run_id}/eval_results_{'_'.join(tags)}.txt",
                            "a",
                        ) as f:
                            f.write(f"{eval_set} batch {batch_idx} WER: {wer}\n")

            avg_wer = jiwer.wer(references, hypotheses) * 100

            if rank == 0:
                with open(
                    f"logs/training/{exp_name}/{run_id}/eval_results_{'_'.join(tags)}.txt",
                    "a",
                ) as f:
                    f.write(f"{eval_set} average WER: {avg_wer}\n")
                wandb.log({f"eval/{eval_set}_wer": avg_wer})

    if rank == 0:
        wandb.log({f"eval_table_{current_step}": eval_table})
        end_time = time.time()
        with open(
            f"logs/training/{exp_name}/{run_id}/epoch_times_{'_'.join(tags)}.txt", "a"
        ) as f:
            f.write(f"eval epoch took {(end_time - start_time) / 60.0} minutes\n")
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
    train_steps: int,
    val_shards: str,
    run_id: Optional[str] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    lr: float = 1.5e-3,
    betas: tuple = (0.9, 0.98),
    eps: float = 1e-6,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    eff_batch_size: int = 256,
    train_batch_size: int = 8,
    val_batch_size: int = 8,
    eval_batch_size: int = 32,
    num_workers: int = 10,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    run_val: bool = True,
    run_eval: bool = False,
) -> None:
    """Main function for training

    Conducts a training loop for the specified number of epochs, with validation and evaluation (if run_eval is True)

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
        epochs: The number of epochs to train for
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

    with open("logs/data/preprocess/shard_to_sample_count.json", "r") as f:
        shard_to_sample_count = json.load(f)

    start_shard, end_shard = train_shards.split("/")[-1].split(".tar")[0][1:-1].split("..")
    train_shard_idx_list = list(range(int(start_shard), int(end_shard) + 1))
    train_samples = sum([shard_to_sample_count[str(shard_idx)] for shard_idx in train_shard_idx_list])
    
    if rank == 0:
        print(f"Training on {train_samples} samples")
    train_dataloader, val_dataloader = prepare_data(
        rank=rank,
        train_shards=train_shards,
        val_shards=val_shards,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        train_samples=train_samples,
        n_text_ctx=n_text_ctx,
        tokenizer=tokenizer,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    # prepare eval dataset
    print(f"Preparing eval sets on rank {rank}")
    eval_sets = ["librispeech_clean", "librispeech_other"]
    eval_loaders = []
    for eval_set in eval_sets:
        eval_dataset = EvalDataset(eval_set=eval_set)

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


    # model instantiation
    if run_id is not None:
        (
            current_step,
            best_val_loss,
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
            rank=rank,
            world_size=world_size,
            train_steps=train_steps,
            train_batch_size=train_batch_size,
            eff_batch_size=eff_batch_size,
        )
    else:
        model = ow.model.Whisper(dims=model_dims).to(rank)
        model = DDP(model, device_ids=[rank], output_device=rank)

        # optimizer and scheduler instantiation
        optimizer = prepare_optim(
            model=model, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

        scheduler, accumulation_steps, warmup_steps, train_steps = prepare_sched(
            train_steps=train_steps,
            world_size=world_size,
            train_batch_size=train_batch_size,
            eff_batch_size=eff_batch_size,
            optimizer=optimizer,
        )

        scaler = GradScaler()

        current_step = 0

    # setting up wandb for logging
    if rank == 0:
        if run_id is None:
            best_val_loss = float("inf")

        run_id, tags, train_res, val_res, train_res_added, val_res_added = setup_wandb(
            run_id=run_id,
            exp_name=exp_name,
            job_type=job_type,
            model_variant=model_variant,
            train_shards=train_shards,
            train_steps=train_steps,
            warmup_steps=warmup_steps,
            accumulation_steps=accumulation_steps,
            world_size=world_size,
            num_workers=num_workers,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            eff_batch_size=eff_batch_size,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
        )
    else:
        if run_id is None:
            best_val_loss = None

    while current_step < train_steps:
        (
            current_step,
            best_val_loss,
            model,
            optimizer,
            scaler,
            scheduler,
            train_res_added,
            val_res_added,
        ) = train(
            rank=rank,
            current_step=current_step,
            train_batch_size=train_batch_size,
            train_dataloader=train_dataloader,
            train_steps=train_steps,
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
            run_val=run_val,
            val_dataloader=val_dataloader,
            model_dims=model_dims,
            model_variant=model_variant,
            best_val_loss=best_val_loss if rank == 0 else None,
            val_res=val_res if rank == 0 else None,
            val_res_added=val_res_added if rank == 0 else None,
            run_eval=run_eval,
            eval_batch_size=eval_batch_size,
            eval_num_workers=num_workers,
            eval_loaders=eval_loaders,
            run_id=run_id if rank == 0 else None,
            tags=tags if rank == 0 else None,
            exp_name=exp_name if rank == 0 else None,
        )

        if rank == 0:
            save_ckpt(
                current_step=current_step,
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

        if run_val:
            print(f"Validation after epoch on rank {rank} at {current_step=} w/ best val loss {best_val_loss}")
            best_val_loss, val_res_added = validate(
                rank=rank,
                current_step=current_step,
                best_val_loss=best_val_loss,
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
                exp_name=exp_name if rank == 0 else None,
                run_id=run_id if rank == 0 else None,
            )

            print(f"Rank {rank} reaching barrier w/ best val loss {best_val_loss}")
            dist.barrier()
            print(f"Rank {rank} passing barrier w/ best val loss {best_val_loss}")

        if run_eval:
            print(f"Evaluation after epoch at {current_step=} on rank {rank}")
            evaluate(
                rank=rank,
                current_step=current_step,
                eval_batch_size=eval_batch_size,
                num_workers=num_workers,
                model=model,
                eval_loaders=eval_loaders,
                normalizer=normalizer,
                tags=tags if rank == 0 else None,
                exp_name=exp_name if rank == 0 else None,
                run_id=run_id if rank == 0 else None,
            )

            print(f"Rank {rank} reaching barrier")
            dist.barrier()
            print(f"Rank {rank} passing barrier")

    cleanup()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    Fire(main)