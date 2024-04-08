from open_whisper import audio, utils
import open_whisper as ow

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast

import os
from dataclasses import dataclass
import numpy as np
import wandb
from typing import List
import time
import jiwer
from whisper.normalizers import EnglishTextNormalizer

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
        n_text_ctx: int,
    ):
        self.audio_files = sorted(audio_files)
        self.transcript_files = sorted(transcript_files)
        self.n_text_ctx = n_text_ctx

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        # not sure if putting it here is bad...
        tokenizer = ow.tokenizer.get_tokenizer(multilingual=False)

        audio_file, audio_input = self.preprocess_audio(self.audio_files[index])
        transcript_file, text_input, text_y, padding_mask = self.preprocess_text(
            self.transcript_files[index], tokenizer
        )

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
        # might need to adjust this to a softer threshold instead of exact match
        # if sum(audio_arr) != 0.0:
        #     mel_spec_normalized = (mel_spec - mel_spec.mean()) / mel_spec.std()
        #     mel_spec_scaled = mel_spec_normalized / (mel_spec_normalized.abs().max())
        return audio_file, mel_spec

    def preprocess_text(self, transcript_file, tokenizer):
        # transcript -> text
        transcript, *_ = utils.TranscriptReader(file_path=transcript_file).read()

        if transcript == {}:
            text_tokens = [tokenizer.no_speech]
        else:
            transcript_text = utils.TranscriptReader.extract_text(transcript)

            text_tokens = tokenizer.encode(transcript_text)

        text_tokens = list(tokenizer.sot_sequence_including_notimestamps) + text_tokens

        text_tokens.append(tokenizer.eot)

        # offset
        text_input = text_tokens[:-1]
        text_y = text_tokens[1:]

        padding_mask = torch.zeros((self.n_text_ctx, self.n_text_ctx))
        padding_mask[:, len(text_input) :] = -float("inf")

        text_input = np.pad(
            text_input,
            pad_width=(0, self.n_text_ctx - len(text_input)),
            mode="constant",
            constant_values=51864,
        )
        text_y = np.pad(
            text_y,
            pad_width=(0, self.n_text_ctx - len(text_y)),
            mode="constant",
            constant_values=51864,
        )

        text_input = torch.tensor(text_input, dtype=torch.long)
        text_y = torch.tensor(text_y, dtype=torch.long)

        return transcript_file, text_input, text_y, padding_mask


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
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def prepare(
    dataset,
    rank,
    world_size,
    batch_size,
    pin_memory=False,
    shuffle=True,
    num_workers=0,
    persistent_workers=False,
):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=42,
        drop_last=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        sampler=sampler,
        persistent_workers=persistent_workers,
    )

    return sampler, dataloader


def cleanup():
    torch.cuda.empty_cache()
    dist.destroy_process_group()


audio_files_train = []
for root, *_ in os.walk("data/audio"):
    if "segments" in root:
        for f in os.listdir(root):
            audio_files_train.append(os.path.join(root, f))

transcript_files_train = []
for root, *_ in os.walk("data/transcripts"):
    if "segments" in root:
        for f in os.listdir(root):
            transcript_files_train.append(os.path.join(root, f))


def main(
    rank,
    world_size,
    model_dims,
    subset=None,
    epochs=50,
    eff_size=256,
    train_batch_size=8,
    val_batch_size=8,
    train_val_split=0.98,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
    persistent_workers=False,
):
    # setup the process groups
    setup(rank, world_size)

    # dataset setup
    tokenizer = ow.tokenizer.get_tokenizer(multilingual=False)
    normalizer = EnglishTextNormalizer()

    if subset is not None:
        rng = np.random.default_rng(seed=42)
        start_idx = rng.choice(range(len(audio_files_train) - subset))

    audio_text_dataset = AudioTextDataset(
        audio_files=(
            audio_files_train
            if subset is None
            else audio_files_train[start_idx : start_idx + subset]
        ),
        transcript_files=(
            transcript_files_train
            if subset is None
            else transcript_files_train[start_idx : start_idx + subset]
        ),
        n_text_ctx=model_dims.n_text_ctx,
    )

    train_size = int(train_val_split * len(audio_text_dataset))
    val_size = len(audio_text_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        audio_text_dataset, [train_size, val_size], generator=generator
    )

    # prepare the dataloaders
    train_sampler, train_dataloader = prepare(
        dataset=train_dataset,
        rank=rank,
        world_size=world_size,
        batch_size=train_batch_size,
        pin_memory=pin_memory,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    val_sampler, val_dataloader = prepare(
        dataset=val_dataset,
        rank=rank,
        world_size=world_size,
        batch_size=val_batch_size,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )

    model = ow.model.Whisper(dims=model_dims).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

    lr = 1.5e-3
    betas = (0.9, 0.98)
    eps = 1e-6
    weight_decay = 0.1
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    epochs = epochs
    accumulation_steps = eff_size // (
        world_size * train_batch_size
    )  # Number of steps over which to accumulate gradients
    total_steps = int(np.ceil(len(train_dataloader) / accumulation_steps) * epochs)
    warmup_steps = np.ceil(0.002 * total_steps)

    if rank == 0:
        best_val_loss = float("inf")
        print(f"{len(train_dataloader)=}")
        print(f"{len(val_dataloader)=}")

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
            "warmup_steps": warmup_steps,
            "train_val_split": train_val_split,
            "num_workers": num_workers,
        }

        tags = [
            "tiny-en",
            "ddp-train",
            "grad-acc",
            "fp16",
            f"subset={subset}" if subset is not None else "subset=full",
            f"lr={lr}",
            f"batch_size={train_batch_size}",
            f"workers={num_workers}",
            f"epochs={epochs}",
            f"train_val_split={train_val_split}",
        ]

        if debug:
            tags.append("debug")

        wandb.init(
            project="open_whisper",
            entity="open-whisper-team",
            config=config,
            save_code=True,
            job_type="training",
            tags=(tags),
            dir="scripts/training",
        )

        train_res = wandb.Artifact("train_res", type="results")
        train_res_added = False
        val_res = wandb.Artifact("val_res", type="results")
        val_res_added = False

    scaler = GradScaler()

    for epoch in range(epochs):
        batch_pred_text = []
        batch_tgt_text = []
        batch_unnorm_pred_text = []
        batch_audio_files = []
        batch_text_files = []
        # necessary to make shuffling work properly across multiple epochs
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        if rank == 0:
            columns = [
                "audio_file",
                "audio_input",
                "transcript_file",
                "pred_text",
                "unnorm_pred_text (pre padding removal)",
                "unnorm_pred_text",
                "tgt_text",
                "unnorm_tgt_text",
                "substitutions",
                "deletions",
                "insertions",
                "tgt_text_len",
                "wer",
            ]
            train_table = wandb.Table(columns=columns)
            print("Training")
            start_time = time.time()

        for batch_idx, batch in enumerate(train_dataloader):
            model.train()
            with autocast():
                (
                    audio_files,
                    transcript_files,
                    audio_input,
                    text_input,
                    text_y,
                    padding_mask,
                ) = batch

                # for logging purposes
                batch_audio_files.extend(audio_files)
                batch_text_files.extend(transcript_files)

                audio_input = audio_input.to(rank)
                text_input = text_input.to(rank)
                text_y = text_y.to(rank)
                padding_mask = padding_mask.to(rank)

                logits = model(audio_input, text_input, padding_mask)

                train_loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    text_y.view(-1),
                    ignore_index=51864,
                )
                train_loss = (
                    train_loss / accumulation_steps
                )  # normalization of loss (gradient accumulation)

            scaler.scale(train_loss).backward()  # accumulate gradients
            train_loss.detach_()
            total_loss += train_loss

            # alerting if loss is nan
            if rank == 0:
                if torch.isnan(train_loss):
                    text = f"Loss is NaN for {audio_files} at epoch {epoch} and batch {batch_idx}!"
                    wandb.alert(title="NaN Loss", text=text)

            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)

            # collecting data for logging
            microbatch_pred_text = []
            microbatch_unnorm_pred_text = []
            for pred_instance in pred.cpu().numpy():
                pred_instance_text = tokenizer.decode(list(pred_instance))
                microbatch_unnorm_pred_text.append(pred_instance_text)
                pred_instance_text = utils.remove_after_endoftext(pred_instance_text)
                microbatch_pred_text.append(pred_instance_text)
            batch_pred_text.extend(microbatch_pred_text)
            batch_unnorm_pred_text.extend(microbatch_unnorm_pred_text)

            microbatch_tgt_text = []
            for text_y_instance in text_y.cpu().numpy():
                tgt_y_instance_text = tokenizer.decode(list(text_y_instance))
                tgt_y_instance_text = tgt_y_instance_text.split("<|endoftext|>")[0]
                tgt_y_instance_text = tgt_y_instance_text + "<|endoftext|>"
                microbatch_tgt_text.append(tgt_y_instance_text)
            batch_tgt_text.extend(microbatch_tgt_text)

            # after accumulation_steps, update weights
            if ((batch_idx + 1) % accumulation_steps) == 0:
                train_loss_tensor = total_loss.clone()
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                train_loss_all = train_loss_tensor.item() / dist.get_world_size()

                norm_batch_tgt_text = [normalizer(text) for text in batch_tgt_text]
                norm_batch_pred_text = [normalizer(text) for text in batch_pred_text]
                norm_tgt_pred_pairs = list(
                    zip(norm_batch_tgt_text, norm_batch_pred_text)
                )
                # no empty references - for WER calculation
                batch_tgt_text_full = [
                    norm_batch_tgt_text[i]
                    for i in range(len(norm_batch_tgt_text))
                    if len(norm_batch_tgt_text[i]) > 0
                ]
                batch_pred_text_full = [
                    norm_batch_pred_text[i]
                    for i in range(len(norm_batch_pred_text))
                    if len(norm_batch_tgt_text[i]) > 0
                ]

                train_wer = jiwer.wer(
                    reference=batch_tgt_text_full, hypothesis=batch_pred_text_full
                )
                # Use torch.tensor to work with dist.all_reduce
                train_wer_tensor = torch.tensor(train_wer, device=rank)
                # Aggregate WER across all processes
                dist.all_reduce(train_wer_tensor, op=dist.ReduceOp.SUM)
                # Calculate the average WER across all processes
                train_wer_all = train_wer_tensor.item() / dist.get_world_size()

                if rank == 0:
                    print(f"{epoch=}")
                    print(f"step={batch_idx + 1}")
                    print(f"effective step={(batch_idx + 1) // accumulation_steps}")
                    print(f"train_loss: {train_loss_all}")
                    print(f"train_wer: {train_wer_all}")

                    wandb.log(
                        {"train_loss": train_loss_all, "train_wer": train_wer_all}
                    )

                    # every 20 steps
                    if ((batch_idx + 1) % (20 * accumulation_steps)) == 0:
                        with open(
                            f"logs/training/training_results_{'_'.join(tags)}.txt",
                            "a",
                        ) as f:
                            if not train_res_added:  # only once
                                train_res.add_file(
                                    f"logs/training/training_results_{'_'.join(tags)}.txt"
                                )
                                train_res_added = True
                                wandb.log_artifact(train_res)

                            for i, (
                                tgt_text_instance,
                                pred_text_instance,
                            ) in enumerate(
                                norm_tgt_pred_pairs[::8]  # should log just 8 examples
                            ):
                                f.write(f"{epoch=}\n")
                                f.write(
                                    f"effective step={(batch_idx + 1) // accumulation_steps}\n"
                                )
                                f.write(f"text_file={batch_text_files[i * 8]}\n")
                                f.write(f"{pred_text_instance=}\n")
                                f.write(
                                    f"unnorm_pred_text_instance={batch_pred_text[i * 8]}\n"
                                )
                                f.write(f"{tgt_text_instance=}\n")
                                f.write(
                                    f"unnorm_tgt_text_instance={batch_tgt_text[i * 8]}\n\n"
                                )

                                # logging to wandb table after 1000 steps
                                if (
                                    (batch_idx + 1) // (1000 * accumulation_steps)
                                ) == 1:
                                    # logging to wandb table
                                    wer = np.round(
                                        utils.calculate_wer(
                                            (tgt_text_instance, pred_text_instance)
                                        ),
                                        2,
                                    )
                                    subs = 0
                                    dels = 0
                                    ins = 0
                                    if len(tgt_text_instance) == 0:
                                        subs = 0
                                        dels = 0
                                        ins = len(pred_text_instance.split())
                                    else:
                                        measures = jiwer.compute_measures(
                                            tgt_text_instance, pred_text_instance
                                        )
                                        subs = measures["substitutions"]
                                        dels = measures["deletions"]
                                        ins = measures["insertions"]

                                    train_table.add_data(
                                        batch_audio_files[i * 8],
                                        wandb.Audio(
                                            batch_audio_files[i * 8],
                                            sample_rate=16000,
                                        ),
                                        batch_text_files[i * 8],
                                        pred_text_instance,
                                        batch_unnorm_pred_text[i * 8],
                                        batch_pred_text[i * 8],
                                        tgt_text_instance,
                                        batch_tgt_text[i * 8],
                                        subs,
                                        dels,
                                        ins,
                                        len(tgt_text_instance.split()),
                                        wer,
                                    )

                            f.write(f"{train_loss_all=}\n")
                            f.write(f"{train_wer_all=}\n\n")

                            # logging to wandb table after 1000 steps
                            if ((batch_idx + 1) // (1000 * accumulation_steps)) == 1:
                                wandb.log({f"train_table_{epoch}": train_table})

                    # checkpointing for every 250 steps
                    if ((batch_idx + 1) % (250 * accumulation_steps)) == 0:
                        print("Saving checkpoint (every 250 steps)")
                        ddp_checkpoint = {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            # You can also save other items such as scheduler state
                            "scheduler_state_dict": (
                                scheduler.state_dict() if scheduler else None
                            ),
                            "dims": model_dims.__dict__,
                            # Include any other information you deem necessary
                        }

                        non_ddp_checkpoint = {
                            "epoch": epoch,
                            "model_state_dict": model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            # You can also save other items such as scheduler state
                            "scheduler_state_dict": (
                                scheduler.state_dict() if scheduler else None
                            ),
                            "dims": model_dims.__dict__,
                        }

                        torch.save(
                            ddp_checkpoint,
                            f"checkpoints/tiny-en-ddp_250_{'_'.join(tags)}.pt",
                        )
                        torch.save(
                            non_ddp_checkpoint,
                            f"checkpoints/tiny-en-non-ddp_250_{'_'.join(tags)}.pt",
                        )

                # Gradient clipping, if necessary, should be done before optimizer.step()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)  # Only update weights after accumulation_steps
                scaler.update()
                scheduler.step()  # Adjust learning rate based on accumulated steps
                current_lr = optimizer.param_groups[0]["lr"]
                # logging learning rate
                if rank == 0:
                    wandb.log({"learning_rate": current_lr})
                optimizer.zero_grad()  # Reset gradients only after updating weights
                total_loss = 0.0

                batch_pred_text = []
                batch_tgt_text = []
                batch_unnorm_pred_text = []
                batch_audio_files = []
                batch_text_files = []

        # If your dataset size is not a multiple of (batch_size * accumulation_steps)
        # Make sure to account for the last set of batches smaller than accumulation_steps
        if len(train_dataloader) % accumulation_steps != 0:
            train_loss_tensor = total_loss.clone()
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            train_loss_all = train_loss_tensor.item() / dist.get_world_size()

            norm_batch_tgt_text = [normalizer(text) for text in batch_tgt_text]
            norm_batch_pred_text = [normalizer(text) for text in batch_pred_text]
            norm_tgt_pred_pairs = list(zip(norm_batch_tgt_text, norm_batch_pred_text))
            # no empty references - for WER calculation
            batch_tgt_text_full = [
                norm_batch_tgt_text[i]
                for i in range(len(norm_batch_tgt_text))
                if len(norm_batch_tgt_text[i]) > 0
            ]
            batch_pred_text_full = [
                norm_batch_pred_text[i]
                for i in range(len(norm_batch_pred_text))
                if len(norm_batch_tgt_text[i]) > 0
            ]

            train_wer = jiwer.wer(
                reference=batch_tgt_text_full, hypothesis=batch_pred_text_full
            )
            # Use torch.tensor to work with dist.all_reduce
            train_wer_tensor = torch.tensor(train_wer, device=rank)
            # Aggregate WER across all processes
            dist.all_reduce(train_wer_tensor, op=dist.ReduceOp.SUM)
            # Calculate the average WER across all processes
            train_wer_all = train_wer_tensor.item() / dist.get_world_size()

            if rank == 0:
                print(f"last batch")
                print(f"{epoch=}")
                print(f"step={batch_idx + 1}")
                print(f"effective step={((batch_idx + 1) // accumulation_steps) + 1}")
                print(f"train_loss: {train_loss_all}")
                print(f"train_wer: {train_wer_all}")

                wandb.log({"train_loss": train_loss_all, "train_wer": train_wer_all})

                with open(
                    f"logs/training/training_results_{'_'.join(tags)}.txt",
                    "a",
                ) as f:
                    for i, (
                        tgt_text_instance,
                        pred_text_instance,
                    ) in enumerate(
                        norm_tgt_pred_pairs[::8]  # should log just 8 examples
                    ):
                        f.write(f"{epoch=}\n")
                        f.write(
                            f"effective step={((batch_idx + 1) // accumulation_steps) + 1}\n"
                        )
                        f.write(f"{batch_text_files[i * 8]}\n")
                        f.write(f"{pred_text_instance=}\n")
                        f.write(f"unnorm_pred_text_instance={batch_pred_text[i * 8]}\n")
                        f.write(f"{tgt_text_instance=}\n")
                        f.write(f"unnorm_tgt_text_instance={batch_tgt_text[i * 8]}\n\n")

                    f.write(f"{train_loss_all=}\n")
                    f.write(f"{train_wer_all=}\n\n")

            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            if rank == 0:
                wandb.log({"learning_rate": current_lr})
            optimizer.zero_grad()
            total_loss = 0.0

            batch_pred_text = []
            batch_tgt_text = []
            batch_unnorm_pred_text = []
            batch_audio_files = []
            batch_text_files = []

        if rank == 0:
            end_time = time.time()
            with open(f"logs/training/epoch_times_{'_'.join(tags)}.txt", "a") as f:
                f.write(
                    f"train epoch {epoch} took {(end_time - start_time) / 60.0} minutes at effective step {(batch_idx + 1) // accumulation_steps}\n"
                )

        # validation
        val_loss = 0.0
        norm_pred_text = []
        norm_tgt_text = []

        if rank == 0:
            columns = [
                "audio_file",
                "audio_input",
                "transcript_file",
                "pred_text",
                "unnorm_pred_text (pre padding removal)",
                "unnorm_pred_text",
                "tgt_text",
                "unnorm_tgt_text",
                "substitutions",
                "deletions",
                "insertions",
                "tgt_text_len",
                "wer",
            ]
            val_table = wandb.Table(columns=columns)
            print("Validation")
            start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                with autocast():
                    model.eval()
                    (
                        audio_files,
                        transcript_files,
                        audio_input,
                        text_input,
                        text_y,
                        padding_mask,
                    ) = batch

                    audio_input = audio_input.to(rank)
                    text_input = text_input.to(rank)
                    text_y = text_y.to(rank)
                    padding_mask = padding_mask.to(rank)

                    logits = model(audio_input, text_input, padding_mask)

                    batch_val_loss = F.cross_entropy(
                        logits.view(-1, logits.shape[-1]),
                        text_y.view(-1),
                        ignore_index=51864,
                    )

                    val_loss += batch_val_loss

                probs = F.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1)

                batch_pred_text = []
                unnorm_pred_text = []
                for pred_instance in pred.cpu().numpy():
                    pred_instance_text = tokenizer.decode(list(pred_instance))
                    unnorm_pred_text.append(pred_instance_text)
                    pred_instance_text = utils.remove_after_endoftext(
                        pred_instance_text
                    )
                    batch_pred_text.append(pred_instance_text)

                batch_tgt_text = []
                for text_y_instance in text_y.cpu().numpy():
                    tgt_y_instance_text = tokenizer.decode(list(text_y_instance))
                    tgt_y_instance_text = tgt_y_instance_text.split("<|endoftext|>")[0]
                    tgt_y_instance_text = tgt_y_instance_text + "<|endoftext|>"
                    batch_tgt_text.append(tgt_y_instance_text)

                norm_batch_tgt_text = [normalizer(text) for text in batch_tgt_text]
                norm_batch_pred_text = [normalizer(text) for text in batch_pred_text]
                norm_tgt_pred_pairs = list(
                    zip(norm_batch_tgt_text, norm_batch_pred_text)
                )
                # no empty references - for WER calculation
                batch_tgt_text_full = [
                    norm_batch_tgt_text[i]
                    for i in range(len(norm_batch_tgt_text))
                    if len(norm_batch_tgt_text[i]) > 0
                ]
                norm_tgt_text.extend(batch_tgt_text_full)
                batch_pred_text_full = [
                    norm_batch_pred_text[i]
                    for i in range(len(norm_batch_pred_text))
                    if len(norm_batch_tgt_text[i]) > 0
                ]
                norm_pred_text.extend(batch_pred_text_full)

                batch_val_wer = jiwer.wer(reference=batch_tgt_text_full, hypothesis=batch_pred_text_full)

                if rank == 0:
                    print(f"{epoch=}")
                    print(f"val step={batch_idx + 1}")
                    print(f"val_loss by batch: {batch_val_loss}")
                    print(f"val_wer by batch: {batch_val_wer}")

                    # every 10 steps
                    if (batch_idx + 1) % 10 == 0:
                        with open(
                            f"logs/training/val_results_{'_'.join(tags)}.txt", "a"
                        ) as f:
                            for i, (tgt_text_instance, pred_text_instance) in enumerate(
                                norm_tgt_pred_pairs
                            ):
                                if not val_res_added:  # only once
                                    val_res.add_file(
                                        f"logs/training/val_results_{'_'.join(tags)}.txt"
                                    )
                                    val_res_added = True
                                    wandb.log_artifact(val_res)

                                f.write(f"{epoch=}\n")
                                f.write(f"{transcript_files[i]}\n")
                                f.write(f"{pred_text_instance=}\n")
                                f.write(
                                    f"unnorm_pred_text_instance={batch_pred_text[i]=}\n"
                                )
                                f.write(f"{tgt_text_instance=}\n")
                                f.write(
                                    f"unnorm_tgt_text_instance={batch_tgt_text[i]=}\n\n"
                                )

                                # logging to wandb table after 80 steps
                                if (batch_idx + 1) == 80:
                                    wer = np.round(
                                        utils.calculate_wer(
                                            (tgt_text_instance, pred_text_instance)
                                        ),
                                        2,
                                    )
                                    subs = 0
                                    dels = 0
                                    ins = 0
                                    if tgt_text_instance == "":
                                        subs = 0
                                        dels = 0
                                        ins = len(pred_text_instance.split())
                                    else:
                                        measures = jiwer.compute_measures(
                                            tgt_text_instance, pred_text_instance
                                        )
                                        subs = measures["substitutions"]
                                        dels = measures["deletions"]
                                        ins = measures["insertions"]

                                    val_table.add_data(
                                        audio_files[i],
                                        wandb.Audio(audio_files[i], sample_rate=16000),
                                        transcript_files[i],
                                        pred_text_instance,
                                        unnorm_pred_text[i],
                                        batch_pred_text[i],
                                        tgt_text_instance,
                                        batch_tgt_text[i],
                                        subs,
                                        dels,
                                        ins,
                                        len(tgt_text_instance.split()),
                                        wer,
                                    )

                            # logging to wandb table after 80 steps
                            if (batch_idx + 1) == 80:
                                wandb.log({f"val_table_{epoch}": val_table})

                            f.write(f"{batch_val_loss=}\n")
                            f.write(f"{batch_val_wer=}\n\n")

            if rank == 0:
                end_time = time.time()
                with open(f"logs/training/epoch_times_{'_'.join(tags)}.txt", "a") as f:
                    f.write(
                        f"val epoch {epoch} took {(end_time - start_time) / 60.0} minutes\n"
                    )

            val_wer = jiwer.wer(reference=norm_tgt_text, hypothesis=norm_pred_text)
            ave_val_loss = val_loss / len(val_dataloader)

            val_wer_tensor = torch.tensor(val_wer, device=rank)
            dist.all_reduce(val_wer_tensor, op=dist.ReduceOp.SUM)
            val_wer_all = val_wer_tensor.item() / dist.get_world_size()

            ave_val_loss_tensor = ave_val_loss.clone()
            dist.all_reduce(ave_val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss_all = ave_val_loss_tensor.item() / dist.get_world_size()

            if rank == 0:
                print(f"val_loss: {val_loss_all}")
                print(f"val_wer: {val_wer_all}")

                wandb.log({"val_loss": val_loss_all, "val_wer": val_wer_all})

                if val_loss_all < best_val_loss:
                    best_val_loss = val_loss_all

                    ddp_checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        # You can also save other items such as scheduler state
                        "scheduler_state_dict": (
                            scheduler.state_dict() if scheduler else None
                        ),
                        "dims": model_dims.__dict__,
                        # Include any other information you deem necessary
                    }

                    non_ddp_checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        # You can also save other items such as scheduler state
                        "scheduler_state_dict": (
                            scheduler.state_dict() if scheduler else None
                        ),
                        "dims": model_dims.__dict__,
                    }

                    torch.save(
                        ddp_checkpoint, f"checkpoints/tiny-en-ddp_{'_'.join(tags)}.pt"
                    )
                    torch.save(
                        non_ddp_checkpoint,
                        f"checkpoints/tiny-en-non-ddp_{'_'.join(tags)}.pt",
                    )

    cleanup()


if __name__ == "__main__":
    # suppose we have 4 gpus
    torch.cuda.empty_cache()
    world_size = 4 if not debug else 1
    subset = None
    epochs = 25
    eff_size = 256
    train_batch_size = 8
    val_batch_size = 8
    train_val_split = 0.99
    num_workers = 18
    pin_memory = False
    shuffle = True
    persistent_workers = True
    mp.spawn(
        main,
        args=[
            world_size,
            model_dims,
            subset,
            epochs,
            eff_size,
            train_batch_size,
            val_batch_size,
            train_val_split,
            num_workers,
            pin_memory,
            shuffle,
            persistent_workers,
        ],
        nprocs=world_size,
    )
