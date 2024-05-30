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
from open_whisper.config.model_dims import VARIANT_TO_DIMS, ModelDimensions
import open_whisper as ow

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
    audio_path = file_path + ".m4a"
    text_path = file_path + ".srt"
    audio_bytes = sample["npy"]
    text_bytes = sample["srt"]
    audio_arr = decode_audio_bytes(audio_bytes)
    transcript_str = decode_text_bytes(text_bytes)

    return audio_path, audio_arr, text_path, transcript_str

def preprocess_audio(audio_arr: np.ndarray) -> torch.Tensor:
    audio_arr = audio.pad_or_trim(audio_arr)
    mel_spec = audio.log_mel_spectrogram(audio_arr)

    return mel_spec, audio_arr

def preprocess_text(transcript_string: str, tokenizer: whisper.tokenizer.Tokenizer, n_text_ctx: int) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    
def preprocess(sample, n_text_ctx: int):
    tokenizer = get_tokenizer(multilingual=False)
    audio_path, audio_arr, text_path, transcript_str = decode_sample(sample)
    audio_input, padded_audio_arr = preprocess_audio(audio_arr)
    text_input, text_y, padding_mask = preprocess_text(transcript_str, tokenizer, n_text_ctx)

    return audio_path, text_path, padded_audio_arr, audio_input, text_input, text_y, padding_mask

def shuffle_shards(shards: str) -> List[str]:
    start_train_shard, end_train_shard = [int(shard_idx) for shard_idx in shards.split("{")[-1].split("}")[0].split("..")]
    rng = np.random.default_rng(42)
    shards_list = np.array(range(start_train_shard, end_train_shard + 1))
    rng.shuffle(shards_list)
    shuffled_shards_list = [f"data/tars/{shard_idx:08d}.tar" for shard_idx in shards_list]
    
    return shuffled_shards_list

#%%
dataset = wds.WebDataset("data/tars/{000000..000005}.tar")

#%%
for sample in dataset:
    temp_sample = sample
    break

#%%
dataset = wds.WebDataset("data/tars/{000000..000005}.tar").map(lambda sample: preprocess(sample, 448))

#%%
dataloader = DataLoader(dataset, batch_size=1, drop_last=False)
for batch in dataloader:
    audio_input, text_input, text_y, padding_mask = batch
    print(audio_input.shape, text_input.shape, text_y.shape, padding_mask.shape)
    break

# %%
