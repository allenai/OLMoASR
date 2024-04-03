from eval import AudioTextDataset
from typing import Literal, Union
import wandb
from fire import Fire
import torch
from torch.utils.data import DataLoader
from whisper.whisper import load_model, decode, DecodingOptions, transcribe
from open_whisper import utils
import jiwer
import numpy as np


def main(
    batch_size: int,
    num_workers: int,
    persistent_workers: int,
    corpus: Literal["librispeech-other", "librispeech-clean", "librispeech-dev-clean", "artie-bias-corpus"],
    compression_ratio_threshold: float,
    w_fp: str,
):
    tags = [
        f"w_cp={w_fp}",
        f"corpus={corpus}",
        "eval",
        f"compression_ratio_threshold={compression_ratio_threshold}",
        "transcribe",
    ]

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
        "tgt_text",
        "unnorm_pred_text",
        "unnorm_tgt_text",
        "subs",
        "del",
        "ins",
        "tgt_text_len",
        "wer",
    ]
    avg_table = wandb.Table(
        columns=["model", "avg_wer", "avg_subs", "avg_del", "avg_ins"]
    )

    device = torch.device("cuda")

    audio_text_dataset = AudioTextDataset(corpus)
    dataloader = DataLoader(
        audio_text_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=persistent_workers,
    )

    w_model = load_model(w_fp, device, "checkpoints", in_memory=True)
    w_model.eval()
    # options = DecodingOptions()

    w_total_wer = 0.0
    w_total_subs = 0.0
    w_total_del = 0.0
    w_total_ins = 0.0

    with torch.no_grad():
        eval_table = wandb.Table(columns=columns)
        for batch_idx, batch in enumerate(dataloader):
            audio_file, audio_input, text_y = batch

            # audio_input = audio_input.to(device)

            # w_results = decode(w_model, audio_input, options)
            w_results = transcribe(
                w_model,
                audio_file[0],
                compression_ratio_threshold=compression_ratio_threshold,
                condition_on_previous_text=False,
                without_timestamps=False,
            )
            # w_text_pred = [result.text for result in w_results]
            w_text_pred = [w_results["text"]]
            w_unnorm_tgt_pred_pairs = list(zip(text_y, w_text_pred))

            w_tgt_pred_pairs = utils.clean_text(w_unnorm_tgt_pred_pairs, "english")

            with open(
                f"logs/eval/{w_fp}-{corpus}-{compression_ratio_threshold}.txt",
                "a",
            ) as f:
                for i, (tgt, pred) in enumerate(w_tgt_pred_pairs):
                    w_wer = utils.calculate_wer((tgt, pred))
                    w_total_wer += w_wer

                    w_measures = jiwer.compute_measures(tgt, pred)
                    w_subs = w_measures["substitutions"]
                    w_del = w_measures["deletions"]
                    w_ins = w_measures["insertions"]
                    w_total_subs += w_subs
                    w_total_del += w_del
                    w_total_ins += w_ins

                    f.write(f"Audio File: {audio_file[i]}\n")
                    f.write(f"Target: {text_y[i]}\n")
                    f.write(f"Prediction: {w_text_pred[i]}\n")
                    f.write(f"Cleaned Target: {tgt}\n")
                    f.write(f"Cleaned Prediction: {pred}\n")
                    f.write(f"WER: {w_wer}\n\n")

                    eval_table.add_data(
                        audio_file[i],
                        wandb.Audio(audio_file[i], sample_rate=16000),
                        pred,
                        tgt,
                        w_text_pred[i],
                        text_y[i],
                        w_subs,
                        w_del,
                        w_ins,
                        len(tgt.split()),
                        w_wer,
                    )

    w_avg_wer = w_total_wer / len(audio_text_dataset)
    print(f"Average WER for {w_fp}: {w_avg_wer}")

    w_avg_subs = w_total_subs / len(audio_text_dataset)
    w_avg_del = w_total_del / len(audio_text_dataset)
    w_avg_ins = w_total_ins / len(audio_text_dataset)

    avg_table.add_data(w_fp, w_avg_wer, w_avg_subs, w_avg_del, w_avg_ins)

    wandb.log({"avg_table": avg_table})
    wandb.log({"eval_table": eval_table})

    wandb.finish()


if __name__ == "__main__":
    Fire(main)
    # main(
    #     batch_size=32,
    #     num_workers=4,
    #     persistent_workers=True,
    #     corpus="librispeech-clean",
    #     normalizer="english",
    #     temperature=0.0,
    #     w_fp="checkpoints/whisper/tiny-en-whisper.pt",
    # )
