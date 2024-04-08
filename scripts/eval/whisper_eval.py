# from whisper.whisper import load_model, audio
# from whisper.whisper.normalizers import EnglishTextNormalizer
from whisper import load_model, audio, DecodingOptions
from whisper.normalizers import EnglishTextNormalizer
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import Literal
import os
import numpy as np
import random
import jiwer
import wandb
import json
from fire import Fire


class Librispeech:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load(self):
        transcript_files = []
        audio_text = {}
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".txt"):
                    transcript_files.append(os.path.join(root, file))

        for file in sorted(transcript_files):
            with open(file, "r") as f:
                for line in f:
                    audio_codes = line.split(" ")[0].split("-")
                    audio_file = os.path.join(
                        self.root_dir,
                        audio_codes[0],
                        audio_codes[1],
                        f"{audio_codes[0]}-{audio_codes[1]}-{audio_codes[2]}.flac",
                    )
                    audio_text[audio_file] = " ".join(line.split(" ")[1:]).strip()

        return list(audio_text.keys()), list(audio_text.values())


class ArtieBiasCorpus:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load(self):
        audio_files = []
        transcript_texts = []
        with open(os.path.join(self.root_dir, "artie-bias-corpus.tsv"), "r") as f:
            next(f)
            for line in f:
                audio_file = os.path.join(self.root_dir, line.split("\t")[1].strip())
                transcript_text = line.split("\t")[2].strip()
                audio_files.append(audio_file)
                transcript_texts.append(transcript_text)

        return audio_files, transcript_texts


class AudioTextDataset(Dataset):
    def __init__(
        self,
        corpus: Literal["librispeech-other", "librispeech-clean", "artie-bias-corpus"],
    ):
        self.corpus = corpus

        if corpus == "librispeech-clean":
            librispeech = Librispeech("data/eval/test-clean-librispeech/test-clean")
            audio_files, transcript_texts = librispeech.load()
        elif corpus == "librispeech-other":
            librispeech = Librispeech("data/eval/test-other-librispeech/test-other")
            audio_files, transcript_texts = librispeech.load()
        elif corpus == "artie-bias-corpus":
            artie_bias_corpus = ArtieBiasCorpus("data/eval/artie-bias-corpus")
            audio_files, transcript_texts = artie_bias_corpus.load()

        self.audio_files = audio_files
        self.transcript_texts = transcript_texts

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        # not sure if putting it here is bad...
        audio_file, audio_input = self.preprocess_audio(self.audio_files[index])
        text_y = self.transcript_texts[index]
        return (
            audio_file,
            audio_input,
            text_y,
        )

    def preprocess_audio(self, audio_file):
        audio_arr = audio.load_audio(audio_file, sr=16000)
        audio_arr = audio.pad_or_trim(audio_arr)
        mel_spec = audio.log_mel_spectrogram(audio_arr)
        return audio_file, mel_spec


def main(
    w_fp: str,
    corpus: Literal["librispeech-other", "librispeech-clean", "artie-bias-corpus"],
    compression_ratio_threshold: float = 2.4,
):
    device = torch.device("cuda")
    audio_text_dataset = AudioTextDataset(corpus)
    dataloader = DataLoader(
        audio_text_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        persistent_workers=True,
    )

    model = load_model(w_fp, device, "checkpoints", in_memory=True)
    model.eval()

    normalizer = EnglishTextNormalizer()
    total_wer = 0.0

    hypotheses = []
    references = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            audio_file, audio_input, text_y = batch
            audio_input = audio_input.to(device)

            options = DecodingOptions(language="en", without_timestamps=True)

            results = model.decode(audio_input, options=options)  # using default arguments
            pred_text = results[0].text
            norm_pred_text = normalizer(pred_text)
            hypotheses.append(norm_pred_text)
            print(f"{norm_pred_text=}\n")
            tgt_text = text_y[0]
            norm_tgt_text = normalizer(tgt_text)
            references.append(norm_tgt_text)
            print(f"{norm_tgt_text=}\n")
            wer = jiwer.wer(reference=norm_tgt_text, hypothesis=norm_pred_text) * 100
            print(f"{wer=}\n")

            # norm_pred_text = normalizer(pred_text)
            # norm_tgt_text = normalizer(tgt_text)

            # wer = jiwer.wer(reference=norm_tgt_text, hypothesis=norm_pred_text) * 100
            # print(f"{wer=}\n")
            # total_wer += wer

        # avg_wer = total_wer / len(dataloader)
        avg_wer = jiwer.wer(references, hypotheses) * 100
        print(f"Average WER: {avg_wer}")


if __name__ == "__main__":
    main(w_fp="checkpoints/whisper/tiny-en-whisper.pt", corpus="artie-bias-corpus")
