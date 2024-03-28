"""Main entry point for the open_whisper package."""

from open_whisper import audio, model, preprocess, tokenizer, utils, decoding
from open_whisper.model import ModelDimensions, Whisper
from open_whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim
