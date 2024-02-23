from open_whisper import audio, tokenizer, model, utils

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

import os
from dataclasses import dataclass
import numpy as np
import wandb
from typing import List


DEVICE = torch.device("cuda:3")
debug = False

# encoder architecture
n_mels = 80
n_audio_ctx = 1500
n_audio_state = 384
n_audio_head = 6
n_audio_layer = 4

# decoder architecture
n_vocab = 51864
n_text_ctx = 448
n_text_state = 384
n_text_head = 6
n_text_layer = 4


class AudioTextDataset(Dataset):
    def __init__(
        self,
        audio_files: List,
        transcript_files: List,
        tokenizer,
        n_text_ctx: int,
    ):
        self.audio_files = sorted(audio_files)
        self.transcript_files = sorted(transcript_files)
        self.tokenizer = tokenizer
        self.n_text_ctx = n_text_ctx

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_file, audio_input = self.preprocess_audio(self.audio_files[index])
        transcript_file, text_tokens, padding_mask = self.preprocess_text(
            self.transcript_files[index]
        )
        # offset
        text_input = text_tokens[:-1]
        text_y = text_tokens[1:]
        return (
            audio_file,
            transcript_file,
            audio_input,
            text_input,
            text_y,
            padding_mask,
        )

    def preprocess_audio(self, audio_file):
        audio_arr = audio.load_audio(audio_file, sr=16000)
        audio_arr = audio.pad_or_trim(audio_arr)
        mel_spec = audio.log_mel_spectrogram(audio_arr)
        if sum(audio_arr) != 0.0:
            mel_spec_normalized = (mel_spec - mel_spec.mean()) / mel_spec.std()
            mel_spec_scaled = mel_spec_normalized / (mel_spec_normalized.abs().max())
        else:
            mel_spec_scaled = mel_spec
        return audio_file, mel_spec_scaled

    def preprocess_text(self, transcript_file):
        # transcript -> text
        transcript, *_ = utils.TranscriptReader(file_path=transcript_file).read()

        if transcript == {}:
            text_tokens = [self.tokenizer.no_speech]
        else:
            transcript_text = utils.TranscriptReader.extract_text(transcript)

            text_tokens = self.tokenizer.encode(transcript_text)

        text_tokens = (
            list(self.tokenizer.sot_sequence_including_notimestamps) + text_tokens
        )

        text_tokens.append(self.tokenizer.eot)

        # for transcript lines that are longer than context length
        # if len(text_tokens) > self.n_text_ctx:
        #     text_tokens = text_tokens[: self.n_text_ctx - 1] + [self.tokenizer.eot]

        padding_mask = torch.zeros(
            (self.n_text_ctx, self.n_text_ctx)
        )
        padding_mask[:, len(text_tokens) :] = -float("inf")

        text_tokens = np.pad(
            text_tokens,
            pad_width=(0, self.n_text_ctx - len(text_tokens)),
            mode="constant",
            constant_values=51864,
        )

        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        return transcript_file, text_tokens, padding_mask


# dataset setup
tokenizer = tokenizer.get_tokenizer(multilingual=True, language="en", task="transcribe")

audio_files = []
for root, dirs, files in os.walk("data/audio"):
    if "segments" in root:
        for f in os.listdir(root):
            audio_files.append(os.path.join(root, f))

transcript_files = []
for root, dirs, files in os.walk("data/transcripts"):
    if "segments" in root:
        for f in os.listdir(root):
            transcript_files.append(os.path.join(root, f))

subset = 50
audio_text_dataset = AudioTextDataset(
    audio_files=audio_files if subset is None else audio_files[:subset],
    transcript_files=transcript_files if subset is None else transcript_files[:subset],
    tokenizer=tokenizer,
    n_text_ctx=448,
)

train_batch_size = 8
val_batch_size = 8
train_size = int(0.8 * len(audio_text_dataset))
val_size = len(audio_text_dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    audio_text_dataset, [train_size, val_size], generator=generator
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)


# model setup
@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


model_dims = ModelDimensions(
    n_mels=80,
    n_audio_ctx=1500,
    n_audio_state=384,
    n_audio_head=6,
    n_audio_layer=4,
    n_vocab=51864,
    n_text_ctx=448,
    n_text_state=384,
    n_text_head=6,
    n_text_layer=4,
)

model = model.Whisper(dims=model_dims).to(DEVICE)
lr = 1.5e-3
betas = (0.9, 0.98)
eps = 1e-6
weight_decay = 0.1
optimizer = AdamW(
    model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
)

epochs = 50
total_steps = len(train_dataloader) * epochs
warmup_steps = 2048
best_val_loss = float("inf")


def lr_lambda(current_step: int) -> float:
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(
        0.0,
        float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
    )


scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
max_grad_norm = 1.0


# model training
model.train()

config = {
    "lr": lr,
    "betas": betas,
    "eps": eps,
    "weight_decay": weight_decay,
    "batch_size": train_batch_size,
    "epochs": epochs,
    "total_steps": total_steps,
    "warmup_steps": "warmup_steps",
}

tags = [
    "tiny-en",
    f"{subset}" if subset is not None else "full",
    f"{lr}",
    f"{train_batch_size}",
]

if debug:
    tags.append("debug")

wandb.init(
    project="open_whisper",
    entity="huongngo-8",
    config=config,
    save_code=True,
    job_type="training",
    tags=tags,
    dir="scripts/training",
)

columns = ["audio_input", "pred_test", "tgt_text", "wer", "epoch"]
val_table = wandb.Table(columns=columns)

for epoch in range(epochs):
    model.train()
    print(f"Training: {epoch + 1}")
    for batch_idx, batch in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()
        audio_files, transcript_files, audio_input, text_input, text_y, padding_mask = (
            batch
        )
        
        audio_input = audio_input.to(DEVICE)
        text_input = text_input.to(DEVICE)
        text_y = text_y.to(DEVICE)
        padding_mask = padding_mask.to(DEVICE)

        logits = model(audio_input, text_input, padding_mask)
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)

        temp_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), text_y.view(-1), ignore_index=51864
        )

        train_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), text_y.view(-1), ignore_index=51864
        )

        pred_text = []
        for pred_instance in pred.cpu().numpy():
            pred_instance_text = tokenizer.decode(list(pred_instance))
            pred_instance_text = pred_instance_text.rsplit("<|endoftext|>", 1)[0]
            pred_text.append(pred_instance_text)

        tgt_text = []
        for text_y_instance in text_y.cpu().numpy():
            tgt_y_instance_text = tokenizer.decode(list(text_y_instance))
            tgt_y_instance_text = tgt_y_instance_text.split("<|endoftext|>")[0]
            tgt_text.append(tgt_y_instance_text)

        # text normalization
        tgt_pred_pairs = utils.clean_text(list(zip(tgt_text, pred_text)), "english")

        train_wer = utils.average_wer(tgt_pred_pairs)

        print(f"{train_loss=}")
        print(f"{train_wer=}")

        wandb.log({"train_loss": train_loss, "train_wer": train_wer})

        with open(f"logs/training/training_results_{'_'.join(tags)}.txt", "a") as f:
            for tgt_text_instance, pred_text_instance in tgt_pred_pairs[:10:2]:
                f.write(f"{pred_text_instance=}\n")
                f.write(f"{len(pred_text_instance)=}\n")
                f.write(f"{tgt_text_instance=}\n")
                f.write(f"{len(tgt_text_instance)=}\n")
            f.write(f"{train_wer=}\n\n")

        train_loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)  # gradient clipping
        optimizer.step()
        scheduler.step()

    # validation
    if epoch % 10 == 0:
        print("\nValidation")
        val_loss = 0.0
        val_wer = 0.0
        for batch_idx, batch in enumerate(val_dataloader):
            audio_files, _, audio_input, _, text_y, _ = batch
            model.eval()
            decoder_input = torch.full(
                (len(audio_files), 1), tokenizer.sot, dtype=torch.long, device=DEVICE
            )
            generated_sequences = [[] for _ in range(len(audio_files))]
            active = torch.ones(len(audio_files), dtype=torch.bool)

            while active.any():
                with torch.no_grad():
                    logits = model(audio_input, decoder_input[:, : n_text_ctx - 1])
                    probs = F.softmax(logits, dim=-1)
                    # not a 1-dim tensor! grows as decoding continues
                    next_token_pred = torch.argmax(probs, dim=-1)

                    for i in range(len(audio_files)):
                        if active[i] and len(generated_sequences[i]) < n_text_ctx - 1:
                            generated_sequences[i].append(next_token_pred[i][-1].item())
                            if next_token_pred[i][-1].item() == tokenizer.eot:
                                active[i] = False
                        elif (
                            active[i] and len(generated_sequences[i]) == n_text_ctx - 1
                        ):
                            active[i] = False

                    if not active.any():
                        break

                    decoder_input = torch.cat(
                        [decoder_input, next_token_pred[:, -1].unsqueeze(1)], dim=-1
                    )

            batch_val_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                text_y.view(-1),
                ignore_index=51864,
            )

            val_loss += batch_val_loss

            tgt_pred_pairs = [
                (tokenizer.decode(text_y[i]), tokenizer.decode(seq))
                for i, seq in enumerate(generated_sequences)
            ]

            # text normalization
            tgt_pred_pairs = utils.clean_text(tgt_pred_pairs, "english")

            batch_val_wer = utils.average_wer(tgt_pred_pairs)
            val_wer += batch_val_wer

            print(f"{batch_val_loss=}")
            print(f"{batch_val_wer=}")

            with open(f"logs/training/val_results_{'_'.join(tags)}.txt", "a") as f:
                for i, (tgt_text_instance, pred_text_instance) in enumerate(
                    tgt_pred_pairs[:10:2]
                ):
                    f.write(f"{pred_text_instance=}\n")
                    f.write(f"{len(pred_text_instance)=}\n")
                    f.write(f"{tgt_text_instance=}\n")
                    f.write(f"{len(tgt_text_instance)=}\n")

                    # logging to wandb table
                    wer = np.round(
                        utils.calculate_wer((tgt_text_instance, pred_text_instance)), 2
                    )
                    val_table.add_data(
                        wandb.Audio(audio_files[i], sample_rate=16000),
                        pred_text_instance,
                        tgt_text_instance,
                        wer,
                        epoch,
                    )

                f.write(f"{batch_val_wer=}\n\n")

        ave_val_wer = val_wer / len(val_dataloader)
        ave_val_loss = val_loss / len(val_dataloader)

        wandb.log({"val_loss": ave_val_loss, "val_wer": ave_val_wer})

        if ave_val_loss < best_val_loss:
            best_val_loss = ave_val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                # You can also save other items such as scheduler state
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "dims": model_dims.__dict__,
                # Include any other information you deem necessary
            }

            torch.save(checkpoint, f"checkpoints/tiny-en_{'_'.join(tags)}.pt")
        print("\n")

wandb.log({"val_table": val_table})
