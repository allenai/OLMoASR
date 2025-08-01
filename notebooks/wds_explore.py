#%%
import os
import glob
from io import BytesIO
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
from olmoasr.config.model_dims import VARIANT_TO_DIMS, ModelDimensions
import olmoasr as oa

import webdataset as wds
import tempfile

#%%
def decode_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    bytes_io = BytesIO(audio_bytes)
    audio_arr = np.load(bytes_io)

    return audio_arr

def decode_text_bytes(text_bytes: bytes) -> str:
    transcript_str = text_bytes.decode("utf-8")

    return transcript_str

def decode_sample(sample: Dict[str, bytes]) -> Tuple[np.ndarray, str]:
    file_path = os.path.join(sample["__url__"], sample["__key__"])
    audio_path = file_path + ".npy"
    text_path = file_path + ".srt"
    audio_bytes = sample["npy"]
    text_bytes = sample["srt"]
    audio_arr = decode_audio_bytes(audio_bytes)
    transcript_str = decode_text_bytes(text_bytes)

    return audio_path, audio_arr, text_path, transcript_str

def preprocess_audio(audio_arr: np.ndarray) -> torch.Tensor:
    audio_arr = audio_arr.astype(np.float32) / 32768.0
    audio_arr = audio.pad_or_trim(audio_arr)
    mel_spec = audio.log_mel_spectrogram(audio_arr)

    return mel_spec, audio_arr

def preprocess_text(transcript_string: str, tokenizer: whisper.tokenizer.Tokenizer, n_text_ctx: int) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    reader = oa.utils.TranscriptReader(transcript_string=transcript_string, ext="srt")
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
    audio_path, audio_arr, text_path, transcript_str = sample
    audio_input, padded_audio_arr = preprocess_audio(audio_arr)
    text_input, text_y, padding_mask = preprocess_text(transcript_str, tokenizer, n_text_ctx)

    return audio_path, text_path, padded_audio_arr, audio_input, text_input, text_y, padding_mask

# %%
tokenizer = get_tokenizer(multilingual=False)
n_text_ctx = 448

dataset = wds.DataPipeline(
    wds.SimpleShardList("/mmfs1/gscratch/efml/hvn2002/ow_440K_wds/036123.tar"),
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
    wds.batched(8))

#%%
dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False,
        persistent_workers=True,
    )

#%%
for batch in dataloader:
    audio_path, text_path, padded_audio_arr, audio_input, text_input, text_y, padding_mask = batch
    break

#%%
# tested this with dataloader - didn't return just 10 samples per epoch (buggy?)
dataset.with_epoch(10)