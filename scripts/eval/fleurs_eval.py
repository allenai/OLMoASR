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
    fleurs = load_dataset("google/fleurs", "en_us")

    dataloader = DataLoader(
        fleurs["test"],
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        persistent_workers=True,
    )

    # model = load_model(w_fp, device, "checkpoints", in_memory=True)
    model = load_model(name=w_fp, device=device, inference=True, in_memory=True)
    model.eval()

    normalizer = EnglishTextNormalizer()
    total_wer = 0.0

    hypotheses = []
    references = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            root_path, audio_path, text_y = (
                batch["path"],
                batch["audio"]["path"],
                batch["transcription"],
            )
            root_path = "/".join(root_path[0].split("/")[:-1])
            audio_path = audio_path[0]
            audio_arr = audio.load_audio(os.path.join(root_path, audio_path), sr=16000)
            audio_arr = audio.pad_or_trim(audio_arr)
            audio_input = audio.log_mel_spectrogram(audio_arr)
            print(audio_input.shape)
            audio_input = audio_input.to(device)

            options = DecodingOptions(language="en", without_timestamps=True)

            results = model.decode(
                audio_input, options=options
            )  # using default arguments
            pred_text = results.text
            norm_pred_text = normalizer(pred_text)
            hypotheses.append(norm_pred_text)
            print(f"{norm_pred_text=}\n")
            norm_tgt_text = normalizer(text_y[0])
            references.append(norm_tgt_text)
            print(f"{norm_tgt_text=}\n")
            wer = jiwer.wer(reference=norm_tgt_text, hypothesis=norm_pred_text) * 100
            print(f"{wer=}\n")

        # avg_wer = total_wer / len(dataloader)
        avg_wer = jiwer.wer(references, hypotheses) * 100
        print(f"Average WER: {avg_wer}")


if __name__ == "__main__":
    main(
        w_fp="checkpoints/archive/sunny-tree-79/tiny-en-non-ddp_tiny-en_ddp-train_grad-acc_fp16_subset=full_lr=0.0015_batch_size=8_workers=18_epochs=25_train_val_split=0.99_inf.pt",
        corpus="fleurs-en",
    )
