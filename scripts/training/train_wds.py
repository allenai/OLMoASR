#%%
import os
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

import webdataset as wds
import tempfile

#%%
dataset = wds.WebDataset("data/tars/000000.tar")

#%%
for i, sample in enumerate(dataset):
    test_sample = sample
    audio_bytes = sample["m4a"]
    transcript_bytes = sample["srt"]
    if i == 0:
        break

#%%
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

def decode_text_bytes(text_bytes: bytes) -> Dict:
    transcript_str = text_bytes.decode("utf-8")
    transcript_file = bytes_to_file(text_bytes, ".srt")

    return transcript_file

def decode_sample(sample: Dict[str, bytes]) -> Tuple[np.ndarray, str]:
    file_path = os.path.join(sample["__url__"], sample["__key__"])
    audio_bytes = sample["m4a"]
    text_bytes = sample["srt"]
    audio_arr = decode_audio_bytes(audio_bytes)
    transcript_file = decode_text_bytes(text_bytes)

    return file_path, audio_arr, transcript_file

def preprocess_audio(audio_arr: np.ndarray) -> torch.Tensor:
    audio_arr = audio.pad_or_trim(audio_arr)
    mel_spec = audio.log_mel_spectrogram(audio_arr)

    return mel_spec

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
    file_path, audio_arr, transcript_file = decode_sample(sample)
    mel_spec = preprocess_audio(audio_arr)
    text_input, text_y, padding_mask = preprocess_text(transcript_file, tokenizer, n_text_ctx)

    return file_path, audio_input, text_input, text_y, padding_mask
# %%
