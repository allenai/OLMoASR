from open_whisper import audio, tokenizer, preprocess, model, utils
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from dataclasses import dataclass
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from torcheval.metrics import WordErrorRate
import wandb
from typing import List


AUDIO_FILE = "data/audio/eh77AUKedyM/segments/00:00:01.501_00:00:30.071.wav"
TRANSCRIPT_FILE = "data/transcripts/eh77AUKedyM/segments/00:00:01.501_00:00:30.071.txt"
EVAL_DIR = "data/eval/LibriSpeech/test-clean"
DEVICE = torch.device("cuda:0")
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
        device: torch.DeviceObjType,
        n_text_ctx: int,
    ):
        self.audio_files = sorted(audio_files)
        self.transcript_files = sorted(transcript_files)
        self.tokenizer = tokenizer
        self.device = device
        self.n_text_ctx = n_text_ctx

        if len(self.audio_files) != len(self.transcript_files):
            raise ValueError(
                "The number of audio files and transcript files must be the same"
            )

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_file, audio_input = self.preprocess_audio(self.audio_files[index])
        text_tokens, eot_index = self.preprocess_text(
            self.transcript_files[index], index
        )
        # offset
        text_input = text_tokens[:-1]
        text_y = text_tokens[1:]
        return audio_file, audio_input, text_input, text_y, eot_index

    def preprocess_audio(self, audio_file):
        audio_arr = audio.load_audio(audio_file, sr=16000)
        audio_arr = audio.pad_or_trim(audio_arr)
        mel_spec = audio.log_mel_spectrogram(audio_arr, device=self.device)
        mel_spec_normalized = (mel_spec - mel_spec.mean()) / mel_spec.std()
        mel_spec_scaled = mel_spec_normalized / (mel_spec_normalized.abs().max())
        return audio_file, mel_spec_scaled

    def preprocess_text(self, transcript_file, file_index):
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

        text_tokens.append(tokenizer.eot)
        eot_index = len(text_tokens) - 1

        text_tokens = np.pad(
            text_tokens,
            pad_width=(0, self.n_text_ctx - len(text_tokens)),
            mode="constant",
            constant_values=0,
        )

        text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=self.device)
        return text_tokens, eot_index


tokenizer = tokenizer.get_tokenizer(multilingual=True, language="en", task="transcribe")
audio_text_dataset = AudioTextDataset(
    audio_files=[
        os.path.join("data/sanity-check/audio/eh77AUKedyM/segments", segment)
        for segment in os.listdir("data/sanity-check/audio/eh77AUKedyM/segments")
    ],
    transcript_files=[
        os.path.join("data/sanity-check/transcripts/eh77AUKedyM/segments", segment)
        for segment in os.listdir("data/sanity-check/transcripts/eh77AUKedyM/segments")
    ],
    tokenizer=tokenizer,
    device=DEVICE,
    n_text_ctx=448,
)

batch_size = 2
audio_text_dataloader = DataLoader(
    audio_text_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)


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

epochs = 8795
total_steps = len(audio_text_dataloader) * epochs
warmup_steps = 2048


def lr_lambda(current_step: int) -> float:
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(
        0.0,
        float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
    )


scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
max_grad_norm = 1.0
metric = WordErrorRate()

# model random init
# model.eval()
# for batch_idx, batch in enumerate(audio_text_dataloader):
#     audio_files, audio_input, text_input, text_y, eot_index = batch

#     logits = model(audio_input, text_input)
#     probs = F.softmax(logits, dim=-1)
#     pred = torch.argmax(probs, dim=-1)

#     pred_text = [
#         tokenizer.decode(list(pred_instance)) for pred_instance in pred.cpu().numpy()
#     ]
#     tgt_text = [
#         tokenizer.decode(list(text_y_instance[: eot_index[i]]))
#         for i, text_y_instance in enumerate(text_y.cpu().numpy())
#     ]
#     metric.update(pred_text, tgt_text)
#     average_wer = metric.compute().cpu().numpy().item() * 100
#     print(f"{average_wer=}")


model.train()

if not debug:
    config = {
        "lr": lr,
        "betas": betas,
        "eps": eps,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "total_steps": total_steps,
        "warmup_steps": "warmup_steps",
    }
    wandb.init(
        project="open_whisper",
        entity="huongngo-8",
        config=config,
        save_code=True,
        job_type="training",
        tags="tiny-en-overfit",
        dir="scripts/training",
    )

columns = ["audio_input", "pred_test", "tgt_text", "wer", "epoch"]
train_table = wandb.Table(columns=columns)

for epoch in range(epochs):
    for batch_idx, batch in enumerate(audio_text_dataloader):
        # print(f"{scheduler.get_last_lr()=}")
        optimizer.zero_grad()
        audio_files, audio_input, text_input, text_y, eot_index = batch

        logits = model(audio_input, text_input)
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)

        pred_text = [
            tokenizer.decode(list(pred_instance))
            for pred_instance in pred.cpu().numpy()
        ]
        tgt_text = [
            tokenizer.decode(list(text_y_instance[: eot_index[i]]))
            for i, text_y_instance in enumerate(text_y.cpu().numpy())
        ]
        metric.update(pred_text, tgt_text)
        average_wer = metric.compute().cpu().numpy().item() * 100

        with open("logs/training_results.txt", "a") as f:
            f.write(f"{pred_text[0]=}\n")
            f.write(f"{tgt_text[0]=}\n")
            f.write(f"{average_wer=}\n\n")

        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), text_y.view(-1), ignore_index=0
        )
        print(f"{loss=}")

        wandb.log({"loss": loss, "average_wer": average_wer})

        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)  # gradient clipping

        optimizer.step()
        scheduler.step()

    if epoch == 1:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # You can also save other items such as scheduler state
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'dims': model_dims.__dict__,
            # Include any other information you deem necessary
        }

        torch.save(checkpoint, "checkpoints/tiny-en.pt")

    for i in range(len(pred_text)):
        pred_text_instance = pred_text[i]
        tgt_text_instance = tgt_text[i]
        metric.update(pred_text_instance, tgt_text_instance)
        wer = metric.compute().cpu().numpy().item() * 100
        train_table.add_data(
            wandb.Audio(audio_files[i], sample_rate=16000),
            pred_text,
            tgt_text,
            wer,
            epoch,
        )

wandb.log({"train_table": train_table})
