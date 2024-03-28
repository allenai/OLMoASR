# using whisper's decoding function
from open_whisper import audio, utils, load_model, decoding
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import Literal
import os
import numpy as np
import jiwer
import wandb


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
        # might need to adjust this to a softer threshold instead of exact match
        # if sum(audio_arr) != 0.0:
        #     mel_spec_normalized = (mel_spec - mel_spec.mean()) / "mel_spec.std()
        #     mel_spec_scaled = mel_spec_normalized / (mel_spec_normalized.abs().max())
        return audio_file, mel_spec


def main(batch_size, num_workers, persistent_workers, corpus, fp):
    tags = [f"{fp.split('/')[2]}", corpus, "eval"]

    wandb.init(
        project="open_whisper",
        entity="open-whisper-team",
        save_code=True,
        job_type="inference",
        tags=(tags),
        dir="scripts/training",
    )

    columns = [
        "audio_file",
        "audio_input",
        "pred_text",
        "pred_text (whisper)",
        "tgt_text",
        "tgt_text (whisper)",
        "unnorm_pred_text",
        "unnorm_tgt_text",
        "subs",
        "subs (whisper)",
        "del",
        "del (whisper)",
        "ins",
        "ins (whisper)",
        "tgt_text_len",
        "wer",
        "wer (whisper)",
    ]
    eval_table = wandb.Table(columns=columns)

    device = torch.device("cuda")

    audio_text_dataset = AudioTextDataset(corpus)
    dataloader = DataLoader(
        audio_text_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=persistent_workers,
    )

    model = load_model(fp, device=device, inference=True)
    model.eval()
    options = decoding.DecodingOptions(language="en", without_timestamps=True)
    total_wer = 0.0

    for batch_idx, batch in enumerate(dataloader):
        audio_file, audio_input, text_y = batch

        audio_input = audio_input.to(device)

        results = decoding.decode(model, audio_input, options)
        text_pred = [result.text for result in results]
        unnorm_tgt_pred_pairs = list(zip(text_y, text_pred))

        tgt_pred_pairs = utils.clean_text(unnorm_tgt_pred_pairs, "english")

        with open(f"logs/eval/{fp.split('/')[2]}-{corpus}.txt", "a") as f:
            for i, (tgt, pred) in enumerate(tgt_pred_pairs):

                f.write(f"Audio File: {audio_file[i]}\n")
                f.write(f"Target: {unnorm_tgt_pred_pairs[i][0]}\n")

                f.write(f"Prediction: {unnorm_tgt_pred_pairs[i][1]}\n")
                f.write(f"Cleaned Target: {tgt}\n")
                f.write(f"Cleaned Prediction: {pred}\n")

                wer = np.round(utils.calculate_wer((tgt, pred)), 2)
                total_wer += wer

                measures = jiwer.compute_measures(tgt, pred)
                subs = measures["substitutions"]
                dels = measures["deletions"]
                ins = measures["insertions"]

                f.write(f"WER: {wer}\n")
                f.write(f"Substitutions: {subs}\n")
                f.write(f"Deletions: {dels}\n")
                f.write(f"Insertions: {ins}\n\n")

    with open(f"logs/eval/{fp.split('/')[2]}-{corpus}.txt", "a") as f:
        avg_wer = total_wer / len(audio_text_dataset)
        f.write(f"Average WER: {avg_wer}\n")
        print(f"Average WER: {avg_wer}")


if __name__ == "__main__":
    main(
        batch_size=16,
        num_workers=4,
        persistent_workers=True,
        corpus="librispeech-clean",
        fp="checkpoints/archive/comic-cloud-73/tiny-en-non-ddp_tiny-en_ddp-train_grad-acc_fp16_subset=full_lr=0.0015_batch_size=8_workers=18_epochs=25_train_val_split=0.99_inf.pt",
    )
