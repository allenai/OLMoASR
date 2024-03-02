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
import sys
import signal

debug = True

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
        # not sure if putting it here is bad or is better to put in __getitem__
        tokenizer = ow.tokenizer.get_tokenizer(
            multilingual=True, language="en", task="transcribe"
        )

        audio_file, audio_input = self.preprocess_audio(self.audio_files[index])
        transcript_file, text_tokens, padding_mask = self.preprocess_text(
            self.transcript_files[index], tokenizer
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

        # for transcript lines that are longer than context length
        # if len(text_tokens) > self.n_text_ctx:
        #     text_tokens = text_tokens[: self.n_text_ctx - 1] + [self.tokenizer.eot]

        padding_mask = torch.zeros((self.n_text_ctx, self.n_text_ctx))
        padding_mask[:, len(text_tokens) :] = -float("inf")

        text_tokens = np.pad(
            text_tokens,
            pad_width=(0, self.n_text_ctx - len(text_tokens)),
            mode="constant",
            constant_values=51864,
        )

        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        return transcript_file, text_tokens, padding_mask


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
    num_workers=0,
    persistent_workers=False,
):
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
        persistent_workers=persistent_workers,
    )

    return dataloader


def cleanup(dataloader):
    del dataloader
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    sys.exit(0)


def signal_handler(sig, frame):
    print("Signal received, stopping...")
    cleanup()


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
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
):
    # setup the process groups
    setup(rank, world_size)

    # dataset setup
    tokenizer = ow.tokenizer.get_tokenizer(
        multilingual=True, language="en", task="transcribe"
    )

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

    train_size = int(0.8 * len(audio_text_dataset))
    val_size = len(audio_text_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        audio_text_dataset, [train_size, val_size], generator=generator
    )

    # prepare the dataloaders
    train_dataloader = prepare(
        dataset=train_dataset,
        rank=rank,
        world_size=world_size,
        batch_size=train_batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    val_dataloader = prepare(
        dataset=val_dataset, rank=rank, world_size=world_size, batch_size=val_batch_size
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
    total_steps = len(train_dataloader) * epochs
    warmup_steps = 0.0002 * total_steps

    if rank == 0:
        best_val_loss = float("inf")

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

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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
        tags = [
            "tiny-en",
            "ddp-train",
            "grad-acc",
            "fp16",
            f"subset={subset}" if subset is not None else "full",
            f"lr={lr}",
            f"batch_size={train_batch_size}",
            f"workers={num_workers}",
            f"epochs={epochs}",
        ]

        if debug:
            tags.append("debug")

        wandb.init(
            project="open_whisper",
            entity="huongngo-8",
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

    accumulation_steps = eff_size / (
        4 * train_batch_size
    )  # Number of steps over which to accumulate gradients
    scaler = GradScaler()

    try:
        for epoch in range(epochs):
            batch_pred_text = []
            batch_tgt_text = []
            batch_audio_files = []
            batch_text_files = []
            model.train()
            optimizer.zero_grad()

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

                if rank == 0:
                    if torch.isnan(train_loss):
                        text = f"Loss is NaN for {audio_files} at epoch {epoch} and batch {batch_idx}!"
                        wandb.alert(title="NaN Loss", text=text)

                probs = F.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1)

                microbatch_pred_text = []
                for pred_instance in pred.cpu().numpy():
                    pred_instance_text = tokenizer.decode(list(pred_instance))
                    pred_instance_text = pred_instance_text.rsplit("<|endoftext|>", 1)[
                        0
                    ]
                    microbatch_pred_text.append(pred_instance_text)
                batch_pred_text.extend(microbatch_pred_text)

                microbatch_tgt_text = []
                for text_y_instance in text_y.cpu().numpy():
                    tgt_y_instance_text = tokenizer.decode(list(text_y_instance))
                    tgt_y_instance_text = tgt_y_instance_text.split("<|endoftext|>")[0]
                    microbatch_tgt_text.append(tgt_y_instance_text)
                batch_tgt_text.extend(microbatch_tgt_text)

                scaler.scale(train_loss).backward()  # accumulate gradients

                if batch_idx > 0 and ((batch_idx + 1) % accumulation_steps == 0):
                    train_loss_tensor = train_loss.clone()
                    dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                    train_loss_all = train_loss_tensor.item() / dist.get_world_size()

                    # text normalization
                    tgt_pred_pairs = utils.clean_text(
                        list(zip(batch_tgt_text, batch_pred_text)), "english"
                    )

                    train_wer = utils.average_wer(tgt_pred_pairs)
                    # Use torch.tensor to work with dist.all_reduce
                    train_wer_tensor = torch.tensor(train_wer, device=rank)
                    # Aggregate WER across all processes
                    dist.all_reduce(train_wer_tensor, op=dist.ReduceOp.SUM)
                    # Calculate the average WER across all processes
                    train_wer_all = train_wer_tensor.item() / dist.get_world_size()

                    if rank == 0:
                        print(f"{epoch=}")
                        print(f"train_loss: {train_loss_all}")
                        print(f"train_wer: {train_wer_all}")

                        wandb.log(
                            {"train_loss": train_loss_all, "train_wer": train_wer_all}
                        )

                        with open(
                            f"logs/training/training_results_{'_'.join(tags)}.txt", "a"
                        ) as f:
                            if not train_res_added:  # only once
                                train_res.add_file(
                                    f"logs/training/training_results_{'_'.join(tags)}.txt"
                                )
                                train_res_added = True
                                wandb.log_artifact(train_res)

                            if (batch_idx + 1) % (50 * accumulation_steps) == 0:
                                columns = [
                                    "audio_file",
                                    "audio_input",
                                    "transcript_file",
                                    "pred_text",
                                    "tgt_text",
                                    "wer",
                                    "epoch",
                                ]
                                train_table = wandb.Table(columns=columns)

                            # rng = np.random.default_rng(seed=42)
                            # pairs_sample = rng.choice(
                            #     tgt_pred_pairs, size=10, replace=False
                            # )

                            for i, (tgt_text_instance, pred_text_instance) in enumerate(
                                tgt_pred_pairs[::4]
                            ):
                                f.write(f"{pred_text_instance=}\n")
                                f.write(f"{len(pred_text_instance)=}\n")
                                f.write(f"{tgt_text_instance=}\n")
                                f.write(f"{len(tgt_text_instance)=}\n")

                                if (batch_idx + 1) % (50 * accumulation_steps) == 0:
                                    # logging to wandb table
                                    wer = np.round(
                                        utils.calculate_wer(
                                            (tgt_text_instance, pred_text_instance)
                                        ),
                                        2,
                                    )
                                    train_table.add_data(
                                        batch_audio_files[i * 4],
                                        wandb.Audio(
                                            batch_audio_files[i * 4], sample_rate=16000
                                        ),
                                        batch_text_files[i * 4],
                                        pred_text_instance,
                                        tgt_text_instance,
                                        wer,
                                        epoch,
                                    )

                            if (batch_idx + 1) % (50 * accumulation_steps) == 0:
                                wandb.log(
                                    {f"train_table_{epoch}_{batch_idx + 1}": train_table}
                                )

                            f.write(f"{train_wer_all=}\n\n")

                        # checkpointing for every 250 batches
                        if ((batch_idx + 1) % (250 * accumulation_steps)) == 0:
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
                    scaler.step(
                        optimizer
                    )  # Only update weights after accumulation_steps
                    scaler.update()
                    scheduler.step()  # Adjust learning rate based on accumulated steps
                    optimizer.zero_grad()  # Reset gradients only after updating weights

                    batch_pred_text = []
                    batch_tgt_text = []
                    batch_audio_files = []
                    batch_text_files = []

            # If your dataset size is not a multiple of (batch_size * accumulation_steps)
            # Make sure to account for the last set of batches smaller than accumulation_steps
            if len(train_dataloader) % accumulation_steps != 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                batch_pred_text = []
                batch_tgt_text = []

            # validation
            val_loss = 0.0
            val_wer = 0.0
            for batch_idx, batch in enumerate(val_dataloader):
                with autocast():
                    model.eval()
                    audio_files, transcript_files, audio_input, _, text_y, _ = batch
                    decoder_input = torch.full(
                        (len(audio_files), 1),
                        tokenizer.sot,
                        dtype=torch.long,
                        device=rank,
                    )
                    generated_sequences = [[] for _ in range(len(audio_files))]
                    active = torch.ones(len(audio_files), dtype=torch.bool)

                    audio_input = audio_input.to(rank)
                    decoder_input = decoder_input.to(rank)
                    text_y = text_y.to(rank)

                    while active.any():
                        with torch.no_grad():
                            logits = model(
                                audio_input, decoder_input[:, : n_text_ctx - 1]
                            )
                            probs = F.softmax(logits, dim=-1)
                            # not a 1-dim tensor! grows as decoding continues
                            next_token_pred = torch.argmax(probs, dim=-1)

                            for i in range(len(audio_files)):
                                if (
                                    active[i]
                                    and len(generated_sequences[i]) < n_text_ctx - 1
                                ):
                                    generated_sequences[i].append(
                                        next_token_pred[i][-1].item()
                                    )
                                    if next_token_pred[i][-1].item() == tokenizer.eot:
                                        active[i] = False
                                elif (
                                    active[i]
                                    and len(generated_sequences[i]) == n_text_ctx - 1
                                ):
                                    active[i] = False

                            if not active.any():
                                break

                            decoder_input = torch.cat(
                                [decoder_input, next_token_pred[:, -1].unsqueeze(1)],
                                dim=-1,
                            )

                    # case where the model predicts context length (e.g hits eot token) less than maximum (because target always maximum w/ padding)
                    # not sure if this is correct because this might not calculate the loss correctly for the case where the model should predict more but doesn't
                    # and doesn't show the penalty the model gets for not predicting more tokens w.r.t the target
                    if logits.shape[1] != text_y.shape[1]:
                        text_y = text_y[
                            :, : logits.shape[1]
                        ]  # mask out remaining tokens so that target and prediction have same context length

                    batch_val_loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        text_y.reshape(-1),
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

                if rank == 0:
                    print(f"val_loss by batch: {batch_val_loss}")
                    print(f"val_wer by batch: {batch_val_wer}")

                    with open(
                        f"logs/training/val_results_{'_'.join(tags)}.txt", "a"
                    ) as f:
                        # rng = np.random.default_rng(seed=42)
                        # pairs_sample = rng.choice(tgt_pred_pairs, size=10, replace=False)

                        for i, (tgt_text_instance, pred_text_instance) in enumerate(
                            tgt_pred_pairs
                        ):
                            if not val_res_added:  # only once
                                val_res.add_file(
                                    f"logs/training/val_results_{'_'.join(tags)}.txt"
                                )
                                val_res_added = True
                                wandb.log_artifact(val_res)

                            columns = [
                                "audio_file",
                                "audio_input",
                                "transcript_file",
                                "pred_text",
                                "tgt_text",
                                "wer",
                                "epoch",
                            ]
                            val_table = wandb.Table(columns=columns)

                            f.write(f"{pred_text_instance=}\n")
                            f.write(f"{len(pred_text_instance)=}\n")
                            f.write(f"{tgt_text_instance=}\n")
                            f.write(f"{len(tgt_text_instance)=}\n")

                            # logging to wandb table
                            wer = np.round(
                                utils.calculate_wer(
                                    (tgt_text_instance, pred_text_instance)
                                ),
                                2,
                            )
                            val_table.add_data(
                                audio_files[i],
                                wandb.Audio(audio_files[i], sample_rate=16000),
                                transcript_files[i],
                                pred_text_instance,
                                tgt_text_instance,
                                wer,
                                epoch,
                            )

                        wandb.log({f"val_table_{epoch}_{batch_idx}": val_table})
                        f.write(f"{batch_val_wer=}\n\n")

            ave_val_wer = val_wer / len(val_dataloader)
            ave_val_loss = val_loss / len(val_dataloader)

            ave_val_wer_tensor = torch.tensor(ave_val_wer, device=rank)
            dist.all_reduce(ave_val_wer_tensor, op=dist.ReduceOp.SUM)
            val_wer_all = ave_val_wer_tensor.item() / dist.get_world_size()

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
    finally:
        cleanup()


if __name__ == "__main__":
    # suppose we have 4 gpus
    torch.cuda.empty_cache()
    world_size = 4
    subset = None
    epochs = 400
    eff_size = 256
    train_batch_size = 8
    val_batch_size = 8
    num_workers = 24
    pin_memory = False
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
            num_workers,
            pin_memory,
            persistent_workers,
        ],
        nprocs=world_size,
    )
