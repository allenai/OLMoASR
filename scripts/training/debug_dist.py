from open_whisper import audio, utils
import open_whisper as ow

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import os
from dataclasses import dataclass
import numpy as np
import wandb
from typing import List
import jiwer


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

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_file, audio_input = self.preprocess_audio(self.audio_files[index])
        text_tokens, padding_mask = self.preprocess_text(self.transcript_files[index])
        # offset
        text_input = text_tokens[:-1]
        text_y = text_tokens[1:]
        return audio_file, audio_input, text_input, text_y, padding_mask

    def preprocess_audio(self, audio_file):
        audio_arr = audio.load_audio(audio_file, sr=16000)
        audio_arr = audio.pad_or_trim(audio_arr)
        mel_spec = audio.log_mel_spectrogram(audio_arr, device=self.device)
        mel_spec_normalized = (mel_spec - mel_spec.mean()) / mel_spec.std()
        mel_spec_scaled = mel_spec_normalized / (mel_spec_normalized.abs().max())
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

        padding_mask = torch.zeros(
            (self.n_text_ctx, self.n_text_ctx), device=self.device
        )
        padding_mask[:, len(text_tokens) :] = -float("inf")

        text_tokens = np.pad(
            text_tokens,
            pad_width=(0, self.n_text_ctx - len(text_tokens)),
            mode="constant",
            constant_values=0,
        )

        text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=self.device)
        return text_tokens, padding_mask


class AudioTextEval(AudioTextDataset):
    def __init__(
        self,
        audio_files: List,
        transcript_files: List,
        tokenizer,
        device: torch.DeviceObjType,
        n_text_ctx: int,
    ):
        super().__init__(audio_files, transcript_files, tokenizer, device, n_text_ctx)

        self.transcript_texts = []
        for file in transcript_files:
            with open(file, "r") as f:
                transcript_text = [
                    (line.split(" ")[0], " ".join(line.split(" ")[1:]).strip())
                    for line in f
                ]
            self.transcript_texts.extend(transcript_text)

    def __getitem__(self, index):
        audio_file, audio_input = self.preprocess_audio(self.audio_files[index])
        text_tokens = self.preprocess_text(*self.transcript_texts[index])

        return audio_file, audio_input, text_tokens[1:]

    def preprocess_text(self, text_id, transcript_text):
        text_tokens = self.tokenizer.encode(transcript_text)

        text_tokens = (
            list(self.tokenizer.sot_sequence_including_notimestamps) + text_tokens
        )

        text_tokens.append(self.tokenizer.eot)

        text_tokens = np.pad(
            text_tokens,
            pad_width=(0, self.n_text_ctx - len(text_tokens)),
            mode="constant",
            constant_values=0,
        )

        text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=self.device)
        return text_id, text_tokens


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


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def prepare(dataset, rank, world_size, batch_size, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        sampler=sampler,
    )

    return dataloader


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, model_dims, train_batch_size=256, val_batch_size=128):
    # setup the process groups
    setup(rank, world_size)

    # dataset setup
    tokenizer = ow.tokenizer.get_tokenizer(
        multilingual=True, language="en", task="transcribe"
    )

    audio_files_train = []
    for root, dirs, files in os.walk("data/audio"):
        if "segments" in root:
            for f in os.listdir(root):
                audio_files_train.append(os.path.join(root, f))

    transcript_files_train = []
    for root, dirs, files in os.walk("data/transcripts"):
        if "segments" in root:
            for f in os.listdir(root):
                transcript_files_train.append(os.path.join(root, f))

    audio_text_dataset = AudioTextDataset(
        audio_files=audio_files_train,
        transcript_files=transcript_files_train,
        tokenizer=tokenizer,
        device=DEVICE,
        n_text_ctx=448,
    )

    data_dirs_val = []
    for root, dirs, files in os.walk("data/eval/LibriSpeech/test-clean"):
        if len(root.split("/")) == 6:
            data_dirs_val.append(root)

    transcript_files_val = []
    audio_files_val = []

    for d in data_dirs_val:
        for f in os.listdir(d):
            if f.endswith("txt"):
                transcript_files_val.append(os.path.join(d, f))
            else:
                audio_files_val.append(os.path.join(d, f))

    audio_text_dataset_val = AudioTextEval(
        audio_files=sorted(audio_files_val),
        transcript_files=sorted(transcript_files_val),
        tokenizer=tokenizer,
        device=DEVICE,
        n_text_ctx=448,
    )

    # prepare the dataloader
    audio_text_dataloader = prepare(
        audio_text_dataset, rank, world_size, train_batch_size
    )
    audio_text_dataloader_val = prepare(
        audio_text_dataset_val, rank, world_size, val_batch_size
    )

    model = ow.model.Whisper(dims=model_dims).to(rank)
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(
        model, device_ids=[rank], output_device=rank, find_unused_parameters=True
    )
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
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    max_grad_norm = 1.0

    # model training
    model.train()

    if rank == 0:
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

        wandb.init(
            project="open_whisper",
            entity="huongngo-8",
            config=config,
            save_code=True,
            job_type="training",
            tags=["tiny-en-overfit"] if not debug else ["tiny-en-overfit-debug"],
            dir="scripts/training",
        )

        columns = ["audio_input", "pred_test", "tgt_text", "wer", "epoch"]
        val_table = wandb.Table(columns=columns)

    for epoch in range(epochs):
        # training
        audio_text_dataloader.sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(audio_text_dataloader):
            optimizer.zero_grad()
            audio_files, audio_input, text_input, text_y, padding_mask = batch

            logits = model(audio_input, text_input, padding_mask)
            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)

            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), text_y.view(-1), ignore_index=0
            )
            loss_tensor = torch.tensor(loss, device=rank)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_agg = loss_tensor.item() / dist.get_world_size()

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

            average_wer = utils.average_wer(tgt_pred_pairs)
            # Use torch.tensor to work with dist.all_reduce
            average_wer_tensor = torch.tensor(average_wer, device=rank)
            # Aggregate WER across all processes
            dist.all_reduce(average_wer_tensor, op=dist.ReduceOp.SUM)
            # Calculate the average WER across all processes
            average_wer = average_wer_tensor.item() / dist.get_world_size()

            if rank == 0:
                print(f"{loss_agg=}")
                print(f"{average_wer=}")

                with open(f"logs/training/training_results.txt", "a") as f:
                    for tgt_text_instance, pred_text_instance in tgt_pred_pairs[:10:2]:
                        f.write(f"{pred_text_instance=}\n")
                        f.write(f"{len(pred_text_instance)=}\n")
                        f.write(f"{tgt_text_instance=}\n")
                        f.write(f"{len(tgt_text_instance)=}\n")
                    f.write(f"{average_wer=}\n\n")

                wandb.log({"loss": loss_agg, "average_wer": average_wer})

            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)  # gradient clipping

            optimizer.step()
            scheduler.step()

        # validation
        audio_text_dataloader_val.sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(audio_text_dataloader_val):
            decoder_input = torch.full(
                (val_batch_size, 1), tokenizer.sot, dtype=torch.long, device=DEVICE
            )
            active = torch.ones(val_batch_size, dtype=torch.bool)
            generated_sequences = [[] for _ in range(val_batch_size)]

            with torch.no_grad():
                audio_files, audio_input, text_y = batch
                logits = model(audio_input, decoder_input)
                probs = F.softmax(logits, dim=-1)
                next_token_pred = torch.argmax(probs, dim=-1)

                for i in range(val_batch_size):
                    if active[i] and len(generated_sequences[i]) < n_text_ctx:
                        generated_sequences[i].append(next_token_pred[i].item())
                        if next_token_pred[i].item() == tokenizer.eot:
                            active[i] = False
                    elif len(generated_sequences[i]) == n_text_ctx:
                        active[i] = False

                if not active.any():
                    break

                decoder_input = torch.cat(
                    [decoder_input, next_token_pred.unsqueeze(-1)], dim=-1
                )

            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), text_y.view(-1), ignore_index=-1
            )
            loss_tensor = torch.tensor(loss, device=rank)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss = loss_tensor.item() / dist.get_world_size()

            tgt_pred_pairs = [
                (tokenizer.decode(text_y[i]), tokenizer.decode(seq))
                for i, seq in enumerate(generated_sequences)
            ]
            # text normalization
            tgt_pred_pairs = utils.clean_text(tgt_pred_pairs, "english")

            average_wer = utils.average_wer(tgt_pred_pairs)
            # Use torch.tensor to work with dist.all_reduce
            average_wer_tensor = torch.tensor(average_wer, device=rank)
            # Aggregate WER across all processes
            dist.all_reduce(average_wer_tensor, op=dist.ReduceOp.SUM)
            # Calculate the average WER across all processes
            average_wer = average_wer_tensor.item() / dist.get_world_size()

            if rank == 0:
                print(f"{loss=}")
                print(f"{average_wer=}")

                with open(f"logs/training/val_results.txt", "a") as f:
                    for i, (tgt_text_instance, pred_text_instance) in enumerate(
                        tgt_pred_pairs[:10:2]
                    ):
                        f.write(f"{pred_text_instance=}\n")
                        f.write(f"{len(pred_text_instance)=}\n")
                        f.write(f"{tgt_text_instance=}\n")
                        f.write(f"{len(tgt_text_instance)=}\n")

                        # logging to wandb table
                        wer = np.round(
                            jiwer.wer(tgt_text_instance, pred_text_instance), 2
                        )
                        val_table.add_data(
                            wandb.Audio(audio_files[i], sample_rate=16000),
                            pred_text,
                            tgt_text,
                            wer,
                            epoch,
                        )

                    f.write(f"{average_wer=}\n\n")

    wandb.log({"val_table": val_table})

    cleanup()


if __name__ == "__main__":
    # suppose we have 4 gpus
    world_size = 4
    mp.spawn(main, args=[world_size, model_dims], nprocs=world_size)
