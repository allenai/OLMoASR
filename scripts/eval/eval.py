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

def main(batch_size, num_workers, persistent_workers, corpus, ow_fp, w_fp):
    tags = [f"{ow_fp.split('/')[2]}", f"{w_fp.split('/')[2]}", corpus, "eval"]

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
        "unnorm_pred_text",
        "unnorm_pred_text (whisper)",
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
    avg_table = wandb.Table(columns=["model", "avg_wer", "avg_subs", "avg_del", "avg_ins"])

    device = torch.device("cuda")

    audio_text_dataset = AudioTextDataset(corpus)
    dataloader = DataLoader(
        audio_text_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=persistent_workers,
    )

    ow_model = load_model(ow_fp, device=device, inference=True)
    w_model = load_model(w_fp, device=device, inference=True)
    ow_model.eval()
    w_model.eval()
    options = decoding.DecodingOptions(language="en", without_timestamps=True)

    ow_total_wer = 0.0
    w_total_wer = 0.0
    ow_total_subs = 0.0
    w_total_subs = 0.0
    ow_total_del = 0.0
    w_total_del = 0.0
    ow_total_ins = 0.0
    w_total_ins = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            eval_table = wandb.Table(columns=columns)
            audio_file, audio_input, text_y = batch

            audio_input = audio_input.to(device)

            ow_results = decoding.decode(ow_model, audio_input, options)
            w_results = decoding.decode(w_model, audio_input, options)
            ow_text_pred = [result.text for result in ow_results]
            w_text_pred = [result.text for result in w_results]
            ow_unnorm_tgt_pred_pairs = list(zip(text_y, ow_text_pred))

            ow_tgt_pred_pairs = utils.clean_text(ow_unnorm_tgt_pred_pairs, "english")
            w_pred = utils.clean_text(w_text_pred, "english")
        
            with open(f"logs/eval/{ow_fp.split('/')[2]}-{corpus}.txt", "a") as f:
                for i, (tgt, pred) in enumerate(ow_tgt_pred_pairs):
                    ow_wer = np.round(utils.calculate_wer((tgt, pred)), 2)
                    w_wer = np.round(utils.calculate_wer((tgt, w_pred[i])), 2)
                    ow_total_wer += ow_wer
                    w_total_wer += w_wer

                    ow_measures = jiwer.compute_measures(tgt, pred)
                    ow_subs = ow_measures["substitutions"]
                    ow_del = ow_measures["deletions"]
                    ow_ins = ow_measures["insertions"]
                    ow_total_subs += ow_subs
                    ow_total_del += ow_del
                    ow_total_ins += ow_ins

                    w_measures = jiwer.compute_measures(tgt, w_pred[i])
                    w_subs = w_measures["substitutions"]
                    w_del = w_measures["deletions"]
                    w_ins = w_measures["insertions"]
                    w_total_subs += w_subs
                    w_total_del += w_del
                    w_total_ins += w_ins

                    f.write(f"Audio File: {audio_file[i]}\n")
                    f.write(f"Target: {text_y[i]}\n")
                    f.write(f"Prediction: {ow_text_pred[i]}\n")
                    f.write(f"Prediction (Whisper): {w_text_pred[i]}\n")
                    f.write(f"Cleaned Target: {tgt}\n")
                    f.write(f"Cleaned Prediction: {pred}\n")
                    f.write(f"Cleaned Prediction (Whisper): {w_pred[i]}\n")
                    f.write(f"WER: {ow_wer}\n")
                    f.write(f"WER (Whisper): {w_wer}\n\n")

                    eval_table.add_data(audio_file[i], 
                                        wandb.Audio(audio_file[i], sample_rate=16000), 
                                        pred,
                                        w_pred[i], 
                                        tgt, 
                                        ow_text_pred[i], 
                                        w_text_pred[i], 
                                        text_y[i], 
                                        ow_subs, 
                                        w_subs, 
                                        ow_del, 
                                        w_del, 
                                        ow_ins, 
                                        w_ins, 
                                        len(tgt.split()), 
                                        ow_wer, 
                                        w_wer)
            
            wandb.log({f"eval_table_{batch_idx + 1}": eval_table})

    ow_avg_wer = ow_total_wer / len(audio_text_dataset)
    w_avg_wer = w_total_wer / len(audio_text_dataset)
    print(f"Average WER for {ow_fp.split('/')[2]}: {ow_avg_wer}")
    print(f"Average WER for {w_fp.split('/')[2]}: {w_avg_wer}")

    ow_avg_subs = ow_total_subs / len(audio_text_dataset)
    w_avg_subs = w_total_subs / len(audio_text_dataset)
    ow_avg_del = ow_total_del / len(audio_text_dataset)
    w_avg_del = w_total_del / len(audio_text_dataset)
    ow_avg_ins = ow_total_ins / len(audio_text_dataset)
    w_avg_ins = w_total_ins / len(audio_text_dataset)

    avg_table.add_data(ow_fp.split("/")[2], ow_avg_wer, ow_avg_subs, ow_avg_del, ow_avg_ins)
    avg_table.add_data(w_fp.split("/")[2], w_avg_wer, w_avg_subs, w_avg_del, w_avg_ins)

    wandb.log({"avg_table": avg_table})

    wandb.finish()


if __name__ == "__main__":
    Fire(main)
    # main(
    #     batch_size=8,
    #     num_workers=4,
    #     persistent_workers=True,
    #     corpus="librispeech-clean",
    #     ow_fp="checkpoints/archive/comic-cloud-73/tiny-en-non-ddp_tiny-en_ddp-train_grad-acc_fp16_subset=full_lr=0.0015_batch_size=8_workers=18_epochs=25_train_val_split=0.99_inf.pt",
    #     w_fp="checkpoints/whisper/tiny-en-whisper.pt",
    # )
