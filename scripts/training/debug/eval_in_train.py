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


def load_ckpt(
    exp_name: str,
    run_id: Optional[str],
    rank: int,
    file_name: Optional[str],
    ckpt_dir: str,
    model_variant: str,
) -> torch.nn.Module:
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

    model.load_state_dict(ckpt["model_state_dict"])

    return model


def evaluate(
    rank: int,
    local_rank: int,
    model: torch.nn.Module,
    eval_dataloader: DataLoader,
    normalizer: EnglishTextNormalizer,
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
    non_ddp_model = model.module
    non_ddp_model.eval()

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

        avg_wer = jiwer.wer(references, hypotheses) * 100

        if rank == 0:
            print(f"average WER: {avg_wer}\n")


def cleanup():
    """Cleanup function for the distributed training"""
    torch.cuda.empty_cache()
    dist.destroy_process_group()


def main(
    model_variant: str,
    exp_name: str,
    ckpt_file_name: Optional[str] = None,
    ckpt_dir: str = "checkpoints",
    eval_dir: str = "data/eval",
    eval_batch_size: Optional[int] = 32,
    num_workers: int = 10,
    pin_memory: bool = True,
    persistent_workers: bool = True,
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
    normalizer = EnglishTextNormalizer()

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = int(os.getenv("RANK", "0"))

    print("Preparing eval sets")
    eval_set = "librispeech-clean"
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

    # model instantiation
    model = load_ckpt(
        exp_name=exp_name,
        run_id=None,
        rank=local_rank,
        file_name=ckpt_file_name,
        ckpt_dir=ckpt_dir,
        model_variant=model_variant,
    )

    evaluate(
        rank=rank,
        local_rank=local_rank,
        model=model,
        eval_dataloader=eval_dataloader,
        normalizer=normalizer,
    )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main(
        model_variant="tiny",
        exp_name="debug_evals_in_train",
        ckpt_file_name="/weka/huongn/checkpoint_00064000_tiny_ddp-train_grad-acc_fp16_non_ddp_inf.pt",
        ckpt_dir="/weka/huongn/ow_ckpts",
        eval_dir="/weka/huongn/ow_eval",
        eval_batch_size=1,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
