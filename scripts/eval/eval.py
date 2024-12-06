import multiprocessing
from typing import Literal, Optional
import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
import jiwer
from whisper import audio, DecodingOptions
from whisper.normalizers import EnglishTextNormalizer
from open_whisper import load_model
from fire import Fire
from tqdm import tqdm
from torchaudio.datasets import TEDLIUM
from scripts.eval.get_eval_set import get_eval_set
from scripts.eval.gen_inf_ckpt import gen_inf_ckpt
import wandb
from tqdm import tqdm

EVAL_SETS = [
    "librispeech_clean",
    "librispeech_other",
    "artie_bias_corpus",
    "fleurs",
    "tedlium",
    "voxpopuli",
    "common_voice",
    "ami_ihm",
    "ami_sdm",
]


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


class Fleurs:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load(self):
        with open(f"{self.root_dir}/test.tsv", "r") as f:
            file_text = [line.split("\t")[1:3] for line in f]
            audio_files, transcript_texts = zip(*file_text)
            audio_files = [f"{self.root_dir}/test/{f}" for f in audio_files]

        return audio_files, transcript_texts


class VoxPopuli:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load(self):
        with open(f"{self.root_dir}/asr_test.tsv", "r") as f:
            next(f)
            file_text = [line.split("\t")[:2] for line in f]
            audio_files, transcript_texts = zip(*file_text)
            audio_files = [f"{self.root_dir}/test/{f}.wav" for f in audio_files]

        return audio_files, transcript_texts


class AMI:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load(self):
        with open(f"{self.root_dir}/text", "r") as f:
            file_text = [line.split(" ", 1) for line in f]
            audio_files, transcript_texts = zip(*file_text)
            audio_files = [
                f"{self.root_dir}/{f.split('_')[1]}/eval_{f.lower()}.wav"
                for f in audio_files
            ]

        return audio_files, transcript_texts


class EvalDataset(Dataset):
    def __init__(
        self,
        eval_set: Literal[
            "librispeech_clean",
            "librispeech_other",
            "artie_bias_corpus",
            "fleurs",
            "tedlium",
            "voxpopuli",
            "common_voice",
            "ami_ihm",
            "ami_sdm",
        ],
        hf_token: Optional[str] = None,
        eval_dir: str = "data/eval",
    ):
        if eval_set == "librispeech_clean":
            root_dir = f"{eval_dir}/librispeech_test_clean"
            if not os.path.exists(root_dir):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

            self.dataset = Librispeech(root_dir=root_dir)
        elif eval_set == "librispeech_other":
            root_dir = f"{eval_dir}/librispeech_test_other"
            if not os.path.exists(root_dir):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

            self.dataset = Librispeech(root_dir=root_dir)
        elif eval_set == "artie_bias_corpus":
            root_dir = f"{eval_dir}/artie-bias-corpus"
            if not os.path.exists(root_dir):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

            self.dataset = ArtieBiasCorpus(root_dir=root_dir)
        elif eval_set == "fleurs":
            root_dir = f"{eval_dir}/fleurs"
            if not os.path.exists(root_dir):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

            self.dataset = Fleurs(root_dir=root_dir)
        elif eval_set == "tedlium":
            if not os.path.exists(f"{eval_dir}/TEDLIUM_release-3"):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

            self.dataset = TEDLIUM(
                root=f"{eval_dir}", release="release3", subset="test"
            )
        elif eval_set == "voxpopuli":
            root_dir = f"{eval_dir}/voxpopuli"
            if not os.path.exists(root_dir):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

            self.dataset = VoxPopuli(root_dir=root_dir)
        elif eval_set == "common_voice":
            if not os.path.exists(f"{eval_dir}/mozilla-foundation___common_voice_5_1"):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir, hf_token=hf_token)

            self.dataset = load_dataset(
                path="mozilla-foundation/common_voice_5_1",
                name="en",
                token=hf_token,
                split="test",
                cache_dir=eval_dir,
                trust_remote_code=True,
                num_proc=15,
                save_infos=True,
            )
        elif eval_set == "ami_ihm":
            root_dir = f"{eval_dir}/ami/ihm"
            if not os.path.exists(root_dir):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

            self.dataset = AMI(root_dir=root_dir)
        elif eval_set == "ami_sdm":
            root_dir = f"{eval_dir}/ami/sdm"
            if not os.path.exists(root_dir):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

            self.dataset = AMI(root_dir=root_dir)

        self.eval_set = eval_set

        if self.eval_set not in ["tedlium", "common_voice"]:
            audio_files, transcript_texts = self.dataset.load()
            self.audio_files = audio_files
            self.transcript_texts = transcript_texts

    def __len__(self):
        if self.eval_set in ["tedlium", "common_voice"]:
            return len(self.dataset)
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_fp = ""
        audio_arr = ""

        if self.eval_set == "tedlium":
            waveform, _, text_y, *_ = self.dataset[index]
            audio_arr = audio.pad_or_trim(waveform[0])
            audio_input = audio.log_mel_spectrogram(audio_arr)
        elif self.eval_set == "common_voice":
            waveform = self.dataset[index]["audio"]["array"]
            sampling_rate = self.dataset[index]["audio"]["sampling_rate"]
            text_y = self.dataset[index]["sentence"]

            if sampling_rate != 16000:
                waveform = librosa.resample(
                    waveform, orig_sr=sampling_rate, target_sr=16000
                )

            audio_arr = audio.pad_or_trim(waveform)
            audio_arr = audio_arr.astype(np.float32)
            audio_input = audio.log_mel_spectrogram(audio_arr)
        else:
            audio_fp = self.audio_files[index]
            audio_input = self.preprocess_audio(audio_fp)
            text_y = self.transcript_texts[index]

        return audio_fp, audio_arr, audio_input, text_y

    def preprocess_audio(self, audio_file):
        audio_arr = audio.load_audio(audio_file, sr=16000)
        audio_arr = audio.pad_or_trim(audio_arr)
        mel_spec = audio.log_mel_spectrogram(audio_arr)
        return mel_spec


def gen_tbl_row(norm_tgt_text_instance, norm_pred_text_instance, audio_input_instance):
    wer = (
        np.round(
            jiwer.wer(
                reference=norm_tgt_text_instance,
                hypothesis=norm_pred_text_instance,
            ),
            2,
        )
        * 100
    )
    measures = jiwer.compute_measures(
        truth=norm_tgt_text_instance, hypothesis=norm_pred_text_instance
    )
    subs = measures["substitutions"]
    dels = measures["deletions"]
    ins = measures["insertions"]

    print(audio_input_instance)
    print(audio_input_instance.numpy())

    return [
        wandb.Audio(audio_input_instance, sample_rate=16000),
        norm_pred_text_instance,
        norm_tgt_text_instance,
        subs,
        dels,
        ins,
        wer,
    ]


def parallel_gen_tbl_row(args):
    return gen_tbl_row(*args)


def main(
    batch_size: int,
    num_workers: int,
    ckpt: str,
    eval_set: Literal[
        "librispeech_clean",
        "librispeech_other",
        "artie_bias_corpus",
        "fleurs",
        "tedlium",
        "voxpopuli",
        "common_voice",
        "ami_ihm",
        "ami_sdm",
    ],
    wandb_log: bool = False,
    wandb_log_dir: str = "wandb",
    eval_dir: str = "data/eval",
    hf_token: Optional[str] = None,
):
    if "inf" not in ckpt:
        ckpt = gen_inf_ckpt(ckpt, ckpt.replace(".pt", "_inf.pt"))

    os.makedirs(wandb_log_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device("cuda")
    dataset = EvalDataset(eval_set=eval_set, hf_token=hf_token, eval_dir=eval_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    model = load_model(name=ckpt, device=device, inference=True, in_memory=True)
    model.eval()

    normalizer = EnglishTextNormalizer()

    hypotheses = []
    references = []

    if wandb_log:
        run_id = wandb.util.generate_id()
        exp_name = f"{eval_set}_eval"
        config = {"ckpt": ckpt.split("/")[-2]}
        wandb_table_cols = [
            "audio",
            "prediction",
            "target",
            "subs",
            "dels",
            "ins",
            "wer",
        ]
        wandb.init(
            id=run_id,
            resume="allow",
            project="open_whisper",
            entity="dogml",
            job_type="evals",
            name=exp_name,
            dir=wandb_log_dir,
            config=config,
            tags=["eval", eval_set],
        )

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            *_, audio_input, text_y = batch

            norm_tgt_text = [normalizer(text) for text in text_y]
            audio_input = audio_input.to(device)

            options = DecodingOptions(language="en", without_timestamps=True)

            results = model.decode(
                audio_input, options=options
            )  # using default arguments

            norm_pred_text = [
                normalizer(results[i].text)
                for i in range(len(results))
                if norm_tgt_text[i] != ""
                and norm_tgt_text[i] != "ignore time segment in scoring"
            ]
            norm_tgt_text = [
                norm_tgt_text[i]
                for i in range(len(results))
                if norm_tgt_text[i] != ""
                and norm_tgt_text[i] != "ignore time segment in scoring"
            ]
            references.extend(norm_tgt_text)
            hypotheses.extend(norm_pred_text)

            if wandb_log:
                # if (batch_idx + 1) // int(np.ceil(len(dataloader) / 10)) == 1:
                # with multiprocessing.Pool() as pool:
                #     eval_table_data = list(
                #         tqdm(
                #             pool.imap_unordered(
                #                 parallel_gen_tbl_row,
                #                 zip(norm_tgt_text, norm_pred_text, audio_input.cpu()),
                #             ),
                #             total=len(norm_tgt_text),
                #         )
                #     )

                # eval_table = wandb.Table(
                #     columns=wandb_table_cols, data=eval_table_data
                # )

                eval_table = wandb.Table(columns=wandb_table_cols)

                for i in range(0, len(norm_pred_text), 8):
                    print(audio_input.cpu()[i])
                    wer = (
                        np.round(
                            jiwer.wer(
                                reference=norm_tgt_text[i],
                                hypothesis=norm_pred_text[i],
                            ),
                            2,
                        )
                        * 100
                    )
                    measures = jiwer.compute_measures(
                        truth=norm_tgt_text[i], hypothesis=norm_pred_text[i]
                    )
                    subs = measures["substitutions"]
                    dels = measures["deletions"]
                    ins = measures["insertions"]

                    eval_table.add_data(
                        wandb.Audio(audio_input.cpu()[i], sample_rate=16000),
                        norm_pred_text[i],
                        norm_tgt_text[i],
                        subs,
                        dels,
                        ins,
                        wer,
                    )

        avg_wer = jiwer.wer(references, hypotheses) * 100
        avg_measures = jiwer.compute_measures(truth=references, hypothesis=hypotheses)
        avg_subs = avg_measures["substitutions"]
        avg_ins = avg_measures["insertions"]
        avg_dels = avg_measures["deletions"]

        print(
            f"Average WER: {avg_wer}, Average Subs: {avg_subs}, Average Ins: {avg_ins}, Average Dels: {avg_dels}"
        )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    Fire(main)
