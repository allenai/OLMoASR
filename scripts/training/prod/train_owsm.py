import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
import glob
import io
import re
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
import zstandard as zstd
import subprocess
from kaldiio import load_mat

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
from olmoasr.config.model_dims import VARIANT_TO_DIMS, ModelDimensions
import olmoasr

from scripts.eval.eval import EvalDataset
from for_logging import TRAIN_TABLE_COLS, EVAL_TABLE_COLS

# Number of examples to log to W&B tables for inspection
WANDB_EXAMPLES = 8

# Set W&B service wait timeout to prevent hanging
os.environ["WANDB__SERVICE_WAIT"] = "300"

# Model parameter counts for different Whisper variants
VARIANT_TO_PARAMS = {
    "tiny": 39 * 10**6,
    "base": 74 * 10**6,
    "small": 244 * 10**6,
    "medium": 769 * 10**6,
    "large": 1550 * 10**6,
}

# Peak FLOPS performance for different GPU hardware (FP16/BF16)
HARDWARE_TO_FLOPS = {"H100": 900 * 10**12, "L40": 366 * 10**12, "A100": 312 * 10**12}


class AudioTextDataset(Dataset):
    """Dataset for OWSM (Open Whisper-style Speech Model) audio and transcript segments.

    Processes audio-text pairs for OWSM training with special OWSM-style text formatting
    including language tags (<eng><asr>) and timestamp tokens. Supports both timestamp
    and no-timestamp training modes with random selection during training.

    Attributes:
        samples: List of sample dictionaries containing audio/transcript metadata
        n_text_ctx: Maximum number of text context tokens
        n_head: Number of attention heads (for mask computation)
    """

    def __init__(
        self,
        samples: List[Dict],
        n_text_ctx: int,
        n_head: int,
    ):
        """Initialize the OWSM dataset.

        Args:
            samples: List of dictionaries with keys 'audio', 'text', 'key'
            n_text_ctx: Maximum text context length for padding/truncation
            n_head: Number of attention heads (used for attention mask creation)
        """
        self.samples = samples
        self.n_text_ctx = n_text_ctx
        self.n_head = n_head

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index) -> Tuple[
        str,
        str,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
        float,
        float,
        float,
    ]:
        """Get a single sample from the OWSM dataset.

        Processes both audio and text for OWSM training, including special handling
        for OWSM text format with language tags and timestamp extraction.

        Args:
            index: Index of the sample to retrieve

        Returns:
            Tuple containing:
            - audio_file: Path to audio file
            - transcript_file: Sample key/identifier
            - padded_audio_arr: Raw audio array (30s)
            - audio_input: Mel spectrogram tensor
            - text_input: Tokenized input sequence
            - text_y: Tokenized target sequence
            - padding_mask: Attention padding mask
            - preproc_time: Total preprocessing time
            - audio_preproc_time: Audio preprocessing time
            - audio_load_time: Audio loading time
            - text_preproc_time: Text preprocessing time
        """
        # Track total preprocessing time for efficiency monitoring
        start_preproc = time.time()
        global tokenizer

        # Extract sample metadata
        sample_dict = self.samples[index]
        audio_file = sample_dict["audio"]
        transcript_string = sample_dict[
            "text"
        ]  # OWSM format: "<eng><asr>text<1.2><3.4>..."
        transcript_file = sample_dict["key"]

        # Extract the final timestamp from OWSM format text (used for audio normalization)
        matches = re.findall(r"<([\d.]+)>", transcript_string)
        norm_end = float(matches[-1]) if matches else 30.0

        # Process text first (may affect audio processing if timestamp mode is used)
        (
            text_input,
            text_y,
            padding_mask,
            timestamp_mode,
            text_preproc_time,
        ) = self.preprocess_text(
            transcript_string,
            transcript_file,
            tokenizer,
            norm_end,
        )

        # If timestamp mode is enabled, use full 30s audio regardless of text end time
        if timestamp_mode is True:
            norm_end = None

        # Process audio with potentially updated norm_end
        audio_input, padded_audio_arr, audio_preproc_time, audio_load_time = (
            self.preprocess_audio(audio_file, norm_end)
        )

        # Calculate total preprocessing time
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
    ) -> Tuple[torch.Tensor, np.ndarray, float, float]:
        """Preprocesses audio data for OWSM training.

        Loads audio using kaldiio (for Kaldi-format files), normalizes to float32,
        pads or trims to 30 seconds, and computes log mel spectrogram.

        Args:
            audio_file: Path to the audio file (Kaldi format)
            norm_end: End time for audio normalization (currently unused but kept for compatibility)

        Returns:
            A tuple containing:
            - mel_spec: Log mel spectrogram tensor
            - audio_arr: Processed audio array (30s at 16kHz)
            - audio_preproc_time: Total audio preprocessing time
            - audio_load_time: Time spent loading the audio file
        """
        start_time = time.time()

        # Load audio using kaldiio (for Kaldi matrix format files)
        sampling_rate, audio_arr = load_mat(audio_file)
        audio_arr = audio_arr.astype(np.float32)
        audio_load_time = time.time() - start_time

        # Pad or trim audio to exactly 30 seconds (480,000 samples at 16kHz)
        audio_arr = audio.pad_or_trim(audio_arr)

        # Compute log mel spectrogram for model input
        mel_spec = audio.log_mel_spectrogram(audio_arr)
        audio_preproc_time = time.time() - start_time

        return mel_spec, audio_arr, audio_preproc_time, audio_load_time

    def preprocess_text(
        self,
        transcript_string: str,
        transcript_file: str,
        tokenizer: whisper.tokenizer.Tokenizer,
        norm_end: Union[int, str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, float]:
        """Preprocesses OWSM-format text data for training.

        Handles OWSM text format with language tags (<eng><asr>) and timestamp tokens.
        Randomly chooses between timestamp and no-timestamp training modes (50/50 split).
        Converts timestamp format from <1.23> to <|1.23|> for Whisper tokenizer compatibility.

        Args:
            transcript_string: OWSM format text: "<eng><asr>Hello world<1.2><3.4>"
            transcript_file: Sample identifier for debugging
            tokenizer: Whisper tokenizer for encoding text
            norm_end: End time in milliseconds (used to detect 30s segments)

        Returns:
            A tuple containing:
            - text_input: Tokenized input sequence tensor
            - text_y: Tokenized target sequence tensor
            - padding_mask: Attention padding mask tensor
            - timestamp_mode: Whether timestamp mode was used
            - text_preproc_time: Time spent preprocessing text
        """
        start_time = time.time()
        timestamp_mode = False

        # Check if this is an empty transcript (only timestamps, no text content)
        text_content = re.sub(
            r"<\d+\.\d+>", "", transcript_string.split("<eng><asr>")[-1]
        ).strip()

        if text_content == "" and norm_end == 30000.0:
            # Empty 30s segment - use no-speech token
            tokens = (
                list(tokenizer.sot_sequence_including_notimestamps)
                + [tokenizer.no_speech]
                + [tokenizer.eot]
            )
        elif np.random.rand() >= 0.5:
            # 50% chance: Use timestamp mode
            # Extract text after language tag: "<eng><asr>text<1.2><3.4>"
            s = transcript_string.split("<eng><asr>")[-1]
            # Convert OWSM timestamp format <1.23> to Whisper format <|1.23|>
            s_modified = re.sub(r"<(\d+\.\d+)>", r"<|\1|>", s)

            # Encode with timestamp tokens allowed
            tokens = tokenizer.encode(
                s_modified, allowed_special=tokenizer.encoding.special_tokens_set
            )
            # Add SOT token, duplicate last timestamp, and EOT token
            tokens = (
                [tokenizer.sot_sequence[0]] + tokens + [tokens[-1]] + [tokenizer.eot]
            )
            timestamp_mode = True
        else:
            # 50% chance: Use no-timestamp mode
            # Extract text after language tag and remove all timestamp tokens
            s = transcript_string.split("<eng><asr>")[-1]
            s_cleaned = re.sub(r"<\d+\.\d+>", "", s)

            # Standard no-timestamp tokenization
            tokens = (
                list(tokenizer.sot_sequence_including_notimestamps)
                + tokenizer.encode(s_cleaned)
                + [tokenizer.eot]
            )

        # Create input/output sequences for teacher forcing (offset by 1)
        text_input = tokens[:-1]  # All tokens except last
        text_y = tokens[1:]  # All tokens except first

        # Validation: Check for sequences exceeding context length
        if len(text_input) > self.n_text_ctx:
            print(
                f"WARNING: text_input length {len(text_input)} exceeds context {self.n_text_ctx}"
            )
            print(f"{transcript_file=}")
            print(f"{timestamp_mode=}")
            print(f"{norm_end=}")
            print(f"{transcript_string=}")
            print(f"{len(text_input)=}")
            print(f"{text_input=}")

        if len(text_y) > self.n_text_ctx:
            print(
                f"WARNING: text_y length {len(text_y)} exceeds context {self.n_text_ctx}"
            )
            print(f"{transcript_file=}")
            print(f"{timestamp_mode=}")
            print(f"{norm_end=}")
            print(f"{transcript_string=}")
            print(f"{len(text_y)=}")
            print(f"{text_y=}")

        # Validation: Check for invalid token indices (should be < vocab size)
        if max(tokens) >= 51864:
            print(f"ERROR: Invalid token index in {transcript_file}")
            print(f"{timestamp_mode=}")
            print(f"{norm_end=}")
            print(f"{transcript_string=}")
            print("Invalid token index found:", max(tokens), "vs max allowed: 51863")

        # Create attention mask to prevent attending to padding tokens
        padding_mask = torch.zeros((self.n_text_ctx, self.n_text_ctx))
        padding_mask[:, len(text_input) :] = -np.inf

        # Note: Causal mask is handled by the model, not added here
        # This is just padding mask to ignore padded positions

        # Pad sequences to context length with padding token (51864)
        text_input = np.pad(
            text_input,
            pad_width=(0, self.n_text_ctx - len(text_input)),
            mode="constant",
            constant_values=51864,  # Padding token index
        )
        text_y = np.pad(
            text_y,
            pad_width=(0, self.n_text_ctx - len(text_y)),
            mode="constant",
            constant_values=51864,  # Padding token index
        )

        # Convert to PyTorch tensors
        text_input = torch.tensor(text_input, dtype=torch.long)
        text_y = torch.tensor(text_y, dtype=torch.long)
        text_preproc_time = time.time() - start_time

        return (
            text_input,
            text_y,
            padding_mask,
            timestamp_mode,
            text_preproc_time,
        )


def init_tokenizer(worker_id: int):
    """Initialize tokenizer in each dataloader worker process.

    Sets up a global tokenizer instance for the worker process. This is called
    automatically by PyTorch's DataLoader for each worker.

    Args:
        worker_id: The ID of the worker process (provided by DataLoader)
    """
    global tokenizer
    tokenizer = get_tokenizer(multilingual=False)


def setup(rank: int) -> None:
    """Initializes the distributed process group for DDP training.

    Sets the CUDA device for the current process and initializes the NCCL
    process group for distributed training with DDP.

    Args:
        rank: The local rank of the current process (GPU device ID)
    """
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")


def open_dicts_file(samples_dicts_file: str) -> List[Dict]:
    """Load sample dictionaries from compressed OWSM dataset files.

    Supports loading JSONL data from either gzip (.gz) or zstandard (.zst)
    compressed files. Each line should contain a JSON object representing
    an OWSM training sample with 'audio', 'text', and 'key' fields.

    Args:
        samples_dicts_file: Path to the compressed JSONL file

    Returns:
        List of dictionaries, each containing OWSM sample metadata
    """
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
) -> Tuple[DataLoader, DistributedSampler]:
    """Prepares the distributed dataloader for DDP training.

    Creates a DistributedSampler and DataLoader configured for distributed training.
    The sampler ensures each process gets a different subset of the data.

    Args:
        dataset: The dataset to create a dataloader for
        batch_size: Number of samples per batch per process
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle: Whether to shuffle data at each epoch
        num_workers: Number of worker processes for data loading
        prefetch_factor: Number of samples loaded in advance by each worker
        persistent_workers: Whether to keep workers alive between epochs

    Returns:
        A tuple containing (dataloader, distributed_sampler)
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
) -> Tuple[DataLoader, DistributedSampler]:
    """Prepares the OWSM training dataset and dataloader for DDP training.

    Creates an AudioTextDataset from OWSM sample dictionaries and sets up a distributed
    dataloader for training. Each sample dictionary should contain 'audio', 'text', and 'key' fields
    in OWSM format.

    Args:
        samples_dicts: List of OWSM sample dictionaries
        train_batch_size: Number of samples per batch per process
        n_text_ctx: Maximum number of text context tokens
        n_head: Number of attention heads (passed to dataset)
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle: Whether to shuffle data at each epoch
        num_workers: Number of worker processes for data loading
        prefetch_factor: Number of samples loaded in advance by each worker
        persistent_workers: Whether to keep workers alive between epochs

    Returns:
        A tuple containing (train_dataloader, train_sampler)
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
    """Prepares the AdamW optimizer for DDP training.

    Creates an AdamW optimizer with the specified hyperparameters for training
    the DDP-wrapped model.

    Args:
        model: The DDP-wrapped model to train
        lr: Learning rate for the optimizer
        betas: Beta parameters for AdamW momentum
        eps: Epsilon value for numerical stability
        weight_decay: L2 regularization weight decay factor

    Returns:
        Configured AdamW optimizer
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
    """Prepares the learning rate scheduler for DDP training.

    Creates a LambdaLR scheduler with linear warmup and cosine decay. Also calculates
    gradient accumulation steps needed to achieve the effective batch size.

    Args:
        train_steps: Total number of training steps
        world_size: The total number of processes (GPUs)
        train_batch_size: The batch size per process
        eff_batch_size: The effective global batch size
        optimizer: The optimizer to attach the scheduler to

    Returns:
        A tuple containing:
        - scheduler: The learning rate scheduler
        - accumulation_steps: Number of gradient accumulation steps
        - warmup_steps: Number of warmup steps (0.2% of total)
        - train_steps: Total number of training steps (unchanged)
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
) -> str:
    """Sets up Weights and Biases logging for DDP OWSM training.

    Initializes W&B with comprehensive training configuration and defines
    custom metrics for tracking training progress, efficiency, and evaluation.

    Args:
        run_id: Existing run ID to resume (None for new run)
        exp_name: Experiment name for organization
        job_type: Type of job for W&B tracking
        model_variant: Whisper model variant being trained
        model_dims: Model dimension configuration
        train_steps: Total number of training steps
        epoch_steps: Number of steps per epoch
        warmup_steps: Number of learning rate warmup steps
        accumulation_steps: Number of gradient accumulation steps
        world_size: Total number of processes (GPUs)
        num_workers: Number of dataloader worker processes
        prefetch_factor: Dataloader prefetch factor
        lr: Learning rate
        betas: AdamW optimizer beta parameters
        eps: AdamW optimizer epsilon value
        weight_decay: L2 regularization weight decay
        eff_batch_size: Effective global batch size
        train_batch_size: Batch size per process
        hardware: Hardware type for FLOPS calculation
        wandb_tags: List of tags for experiment organization

    Returns:
        The W&B run ID (generated if None was provided)
    """
    # Create comprehensive configuration dictionary for W&B tracking
    config = {
        # Optimizer configuration
        "lr": lr,
        "betas": betas,
        "eps": eps,
        "weight_decay": weight_decay,
        # Batch size configuration
        "eff_batch_size": eff_batch_size,
        "train_batch_size": train_batch_size,
        # Training schedule
        "train_steps": train_steps,
        "epoch_steps": epoch_steps,
        "warmup_steps": warmup_steps,
        "accumulation_steps": accumulation_steps,
        # Distributed training
        "world_size": world_size,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        # Model configuration
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
        # Performance metrics
        "model_params": VARIANT_TO_PARAMS[model_variant],
        "peak_flops": HARDWARE_TO_FLOPS[hardware],
    }

    # Generate new run ID if not provided
    if run_id is None:
        run_id = wandb.util.generate_id()

    # Initialize W&B with configuration
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

    # Define custom metrics for better visualization
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

    model = olmoasr.model.OLMoASR(dims=ckpt["dims"]).to(rank)
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


def gen_pred(
    logits: torch.Tensor, text_y: torch.Tensor, tokenizer: whisper.tokenizer.Tokenizer
) -> Tuple[List[str], List[str], List[str]]:
    """Generate predictions from model logits and decode them to text for OWSM.

    Takes model output logits and ground truth targets, converts logits to predictions
    using argmax, then decodes both predictions and targets to text for evaluation.

    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        text_y: Ground truth target tokens of shape (batch_size, seq_len)
        tokenizer: Whisper tokenizer for decoding tokens to text

    Returns:
        A tuple containing:
        - microbatch_pred_text: List of normalized predicted text strings
        - microbatch_unnorm_pred_text: List of raw predicted text (with timestamps)
        - microbatch_tgt_text: List of ground truth text strings
    """
    # Convert logits to predictions via argmax
    probs = F.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)

    # Decode predictions to text
    microbatch_pred_text = []
    microbatch_unnorm_pred_text = []
    for pred_instance in pred.cpu().numpy():
        # Decode with timestamps preserved
        pred_instance_text = tokenizer.decode_with_timestamps(list(pred_instance))
        microbatch_unnorm_pred_text.append(pred_instance_text)
        # Remove content after <|endoftext|> for evaluation
        pred_instance_text = olmoasr.utils.remove_after_endoftext(pred_instance_text)
        microbatch_pred_text.append(pred_instance_text)

    # Decode ground truth targets to text
    microbatch_tgt_text = []
    for text_y_instance in text_y.cpu().numpy():
        # Remove padding tokens (51864) before decoding
        text_y_instance = list(filter(lambda token: token != 51864, text_y_instance))
        tgt_y_instance_text = tokenizer.decode_with_timestamps(list(text_y_instance))
        # Ensure text ends with endoftext token for consistency
        tgt_y_instance_text = tgt_y_instance_text.split("<|endoftext|>")[0]
        tgt_y_instance_text = tgt_y_instance_text + "<|endoftext|>"
        microbatch_tgt_text.append(tgt_y_instance_text)

    return microbatch_pred_text, microbatch_unnorm_pred_text, microbatch_tgt_text


def calc_pred_wer(
    batch_tgt_text: List[str],
    batch_pred_text: List[str],
    normalizer: EnglishTextNormalizer,
) -> Tuple[List[Tuple[str, str]], float, int, int, int]:
    """Calculate Word Error Rate (WER) and error statistics for OWSM predictions.

    Normalizes target and predicted text, then computes WER and detailed error
    counts (substitutions, deletions, insertions). Filters out empty references
    to avoid division by zero in WER calculation.

    Args:
        batch_tgt_text: List of ground truth text strings
        batch_pred_text: List of predicted text strings
        normalizer: Text normalizer for consistent evaluation

    Returns:
        A tuple containing:
        - norm_tgt_pred_pairs: List of (normalized_target, normalized_prediction) pairs
        - train_wer: Word Error Rate as percentage (0-100)
        - subs: Number of substitution errors
        - dels: Number of deletion errors
        - ins: Number of insertion errors
    """
    # Normalize all text for consistent evaluation
    norm_batch_tgt_text = [normalizer(text) for text in batch_tgt_text]
    norm_batch_pred_text = [normalizer(text) for text in batch_pred_text]
    norm_tgt_pred_pairs = list(zip(norm_batch_tgt_text, norm_batch_pred_text))

    # Filter out empty references to avoid WER calculation issues
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

    # Calculate WER and error statistics
    if len(batch_tgt_text_full) == 0 and len(batch_pred_text_full) == 0:
        # All references are empty - perfect match
        train_wer = 0.0
        subs = 0
        dels = 0
        ins = 0
    else:
        # Calculate WER as percentage
        train_wer = (
            jiwer.wer(
                reference=batch_tgt_text_full,
                hypothesis=batch_pred_text_full,
            )
            * 100
        )
        # Get detailed error counts
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
    global_step: int,
    train_table: wandb.Table,
    run_id: str,
    batch_audio_files: List[str],
    batch_audio_arr: List[np.ndarray],
    batch_text_files: List[str],
    batch_pred_text: List[str],
    batch_tgt_text: List[str],
    batch_unnorm_pred_text: List[str],
    norm_tgt_pred_pairs: List[Tuple[str, str]],
) -> None:
    """Log training examples to Weights & Biases table for OWSM training.

    Creates detailed logging entries for training samples including audio,
    predictions, targets, and error statistics. Each sample gets its own
    row in the W&B table for inspection.

    Args:
        global_step: Current training step number
        train_table: W&B table to add data to
        run_id: Unique identifier for the training run
        batch_audio_files: List of audio file paths
        batch_audio_arr: List of audio arrays for W&B Audio objects
        batch_text_files: List of sample keys/identifiers
        batch_pred_text: List of normalized predicted text
        batch_tgt_text: List of original target text
        batch_unnorm_pred_text: List of raw predicted text (with timestamps)
        norm_tgt_pred_pairs: List of (normalized_target, normalized_prediction) pairs
    """
    for i, (
        tgt_text_instance,
        pred_text_instance,
    ) in enumerate(norm_tgt_pred_pairs):
        # Calculate per-sample WER using OWSM utilities
        wer = np.round(
            olmoasr.utils.calculate_wer((tgt_text_instance, pred_text_instance)),
            2,
        )

        # Calculate detailed error statistics
        subs = 0
        dels = 0
        ins = 0
        if len(tgt_text_instance) == 0:
            # Empty reference - only insertions possible
            subs = 0
            dels = 0
            ins = len(pred_text_instance.split())
        else:
            # Non-empty reference - calculate all error types
            measures = jiwer.compute_measures(tgt_text_instance, pred_text_instance)
            subs = measures["substitutions"]
            dels = measures["deletions"]
            ins = measures["insertions"]

        # Add row to W&B table with all relevant information
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

    # Log the complete table to W&B
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
    int, int, int, Optional[float], DDP, torch.optim.Optimizer, GradScaler, LambdaLR
]:
    """Main DDP training loop for OWSM with automatic mixed precision.

    Performs forward pass, backward pass, gradient accumulation, and optional evaluation.
    Includes comprehensive logging of training metrics, timing, and examples to W&B.
    Handles checkpointing and supports both synchronous evaluation and asynchronous evaluation.

    Args:
        rank: Global rank of the current process across all nodes
        local_rank: Local rank within the current node (GPU device ID)
        global_step: Current global training step across all processes
        local_step: Current local step for this process
        train_batch_size: Batch size per process for training
        train_dataloader: DataLoader providing OWSM training batches
        train_sampler: DistributedSampler for coordinating data across processes
        train_steps: Total number of training steps to perform
        epoch_steps: Number of steps per epoch
        epoch: Current epoch number
        scaler: Gradient scaler for mixed precision training
        model: DDP-wrapped OWSM model for distributed training
        tokenizer: Whisper tokenizer for text processing
        normalizer: Text normalizer for evaluation metrics
        optimizer: Optimizer for parameter updates
        scheduler: Learning rate scheduler
        accumulation_steps: Number of gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        model_dims: Model dimension configuration (for checkpointing)
        model_variant: Model variant name (e.g., "base", "large")
        best_eval_wer: Current best evaluation WER (for comparison)
        run_eval: Whether to run evaluation during training
        eval_loaders: List of evaluation dataloaders (if not async)
        run_id: Unique identifier for this training run
        tags: List of tags for experiment tracking
        exp_name: Experiment name for logging and checkpointing
        log_dir: Directory for saving logs and results
        ckpt_dir: Directory for saving model checkpoints
        train_log_freq: Frequency (in steps) for detailed training logging
        eval_freq: Frequency (in steps) for running evaluation
        ckpt_freq: Frequency (in steps) for saving checkpoints
        verbose: Whether to enable verbose model output
        precision: Torch dtype for mixed precision training
        async_eval: Whether to run evaluation asynchronously
        eval_script_path: Path to evaluation script (for async eval)
        eval_dir: Directory containing evaluation datasets
        eval_wandb_log: Whether to log evaluation results to W&B
        eval_batch_size: Batch size for evaluation
        run_id_dir: Directory for storing run IDs
        eval_on_gpu: Whether to run evaluation on GPU

    Returns:
        A tuple containing:
        - global_step: Updated global step count
        - local_step: Updated local step count
        - epoch: Current epoch number
        - best_eval_wer: Updated best evaluation WER (if evaluation was run)
        - model: The trained DDP model
        - optimizer: Updated optimizer state
        - scaler: Updated gradient scaler state
        - scheduler: Updated learning rate scheduler state
    """
    # Initialize lists for collecting training examples for logging
    batch_pred_text = []
    batch_tgt_text = []
    batch_unnorm_pred_text = []
    batch_audio_files = []
    batch_text_files = []
    batch_audio_arr = []

    # Initialize training state
    total_loss = 0.0
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear gradients from previous iteration

    # Initialize W&B table for logging training examples (only on main process)
    if rank == 0:
        train_table = wandb.Table(columns=TRAIN_TABLE_COLS)

    # Set epoch for distributed sampler to ensure proper data shuffling
    train_sampler.set_epoch(epoch)
    start_dl = time.time()  # Start timing data loading

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
    """Launch asynchronous evaluation subprocess for DDP OWSM training.

    Starts evaluation in a separate process to avoid blocking training. Only
    the main process (rank 0) launches the evaluation to prevent duplicate runs.
    """
    # Get environment variables for evaluation process
    wandb_log_dir = os.getenv("WANDB_DIR")
    hf_token = os.getenv("HF_TOKEN")

    # Build command line arguments for evaluation script
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

    # Only rank 0 launches evaluation to avoid duplicates
    if rank == 0:
        subprocess.Popen(cmd)


def cleanup():
    """Cleanup function for distributed DDP training.

    Clears GPU memory cache and destroys the distributed process group
    to properly shut down distributed training.
    """
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
    """Main training function for DDP OWSM model training.

    Orchestrates the complete DDP training pipeline including:
    - Distributed training setup with DDP wrapping
    - Data loading from compressed OWSM JSONL files
    - Model initialization with DDP wrapping
    - Training loop with gradient accumulation and mixed precision
    - Periodic evaluation (sync or async) on LibriSpeech datasets
    - Comprehensive logging to Weights & Biases

    Supports mixed precision training modes and resume functionality from previous runs.
    Uses OWSM-specific text formatting with language tags and timestamp processing.

    Args:
        model_variant: Whisper model variant ("tiny", "base", "small", "medium", "large")
        exp_name: Experiment name for logging and checkpoint organization
        job_type: Type of job for W&B tracking (e.g., "training", "fine-tuning")
        samples_dicts_dir: Directory containing compressed OWSM JSONL files
        train_steps: Total number of training steps to perform
        epoch_steps: Number of steps per epoch (for periodic evaluation/checkpointing)
        ckpt_file_name: Specific checkpoint filename to resume from (optional)
        ckpt_dir: Directory for saving and loading model checkpoints
        log_dir: Directory for saving training logs and results
        eval_dir: Directory containing evaluation datasets
        run_id_dir: Directory for storing unique run identifiers
        lr: Learning rate for AdamW optimizer
        betas: Beta parameters for AdamW optimizer momentum
        eps: Epsilon parameter for AdamW optimizer numerical stability
        weight_decay: L2 regularization weight decay factor
        max_grad_norm: Maximum gradient norm for gradient clipping
        eff_batch_size: Effective global batch size across all processes
        train_batch_size: Batch size per process for training
        eval_batch_size: Batch size per process for evaluation
        num_workers: Number of worker processes for data loading
        prefetch_factor: Number of samples prefetched by each worker
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle: Whether to shuffle training data each epoch
        persistent_workers: Whether to keep workers alive between epochs
        run_eval: Whether to run periodic evaluation during training
        train_log_freq: Frequency (in steps) for detailed training logging
        eval_freq: Frequency (in steps) for running evaluation
        ckpt_freq: Frequency (in steps) for saving checkpoints
        verbose: Whether to enable verbose model output
        precision: Mixed precision mode ("bfloat16", "float16", "float32")
        hardware: Hardware type for FLOPS calculation ("H100", "A100", "L40")
        async_eval: Whether to run evaluation asynchronously (vs blocking)
        eval_script_path: Path to evaluation script (for async evaluation)
        eval_wandb_log: Whether to log evaluation results to W&B
        eval_on_gpu: Whether to run evaluation on GPU (vs CPU)
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

    samples_dicts_files = glob.glob(f"{samples_dicts_dir}/*.jsonl.*")
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
        model = olmoasr.model.OLMoASR(dims=model_dims).to(local_rank)
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
