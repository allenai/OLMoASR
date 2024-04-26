# from whisper.whisper import load_model, audio
# from whisper.whisper.normalizers import EnglishTextNormalizer
from whisper import audio, DecodingOptions
from whisper.normalizers import EnglishTextNormalizer
from open_whisper import load_model
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchaudio.datasets import TEDLIUM
from datasets import load_dataset
from typing import Literal
import librosa
import os
import numpy as np
import random
import jiwer
import wandb
import json
from fire import Fire


def main(
    w_fp: str,
    corpus: Literal[
        "librispeech-other",
        "librispeech-clean",
        "artiebiascorpus",
        "tedlium",
        "fleurs-en",
        "voxpopuli-en",
        "commonvoice",
    ],
):
    device = torch.device("cuda")
    commonvoice = load_dataset(
        "mozilla-foundation/common_voice_5_1",
        "en",
        token="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",
        split="test",
    )

    model = load_model(name=w_fp, device=device, inference=True, in_memory=True)
    model.eval()

    normalizer = EnglishTextNormalizer()
    total_wer = 0.0

    hypotheses = []
    references = []

    with torch.no_grad():
        for d in commonvoice:
            raw_audio = d["audio"]["array"]
            sampling_rate = d["audio"]["sampling_rate"]
            text_y = d["sentence"]

            if sampling_rate != 16000:
                raw_audio = librosa.resample(raw_audio, orig_sr=sampling_rate, target_sr=16000)

            audio_arr = audio.pad_or_trim(raw_audio)
            audio_arr = audio_arr.astype(np.float32)
            audio_input = audio.log_mel_spectrogram(audio_arr)
            audio_input = audio_input.to(device)

            options = DecodingOptions(language="en", without_timestamps=True)

            results = model.decode(
                audio_input, options=options
            )  # using default arguments
            pred_text = results.text
            norm_pred_text = normalizer(pred_text)
            print(f"{norm_pred_text=}\n")
            norm_tgt_text = normalizer(text_y)
            print(f"{norm_tgt_text=}\n")
            if len(norm_tgt_text) == 0:
                wer = 0.0
            else:
                wer = jiwer.wer(reference=norm_tgt_text, hypothesis=norm_pred_text) * 100
                hypotheses.append(norm_pred_text)
                references.append(norm_tgt_text)
            print(f"{wer=}\n")

        avg_wer = jiwer.wer(references, hypotheses) * 100
        print(f"Average WER: {avg_wer}")

if __name__ == "__main__":
    main(
        w_fp="checkpoints/archive/sunny-tree-79/tiny-en-non-ddp_tiny-en_ddp-train_grad-acc_fp16_subset=full_lr=0.0015_batch_size=8_workers=18_epochs=25_train_val_split=0.99_inf.pt",
        corpus="commonvoice",
    )
