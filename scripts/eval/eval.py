from typing import Literal, Optional
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
import jiwer
from whisper import audio, DecodingOptions
from whisper.normalizers import EnglishTextNormalizer
from open_whisper import load_model
import torchaudio
from fire import Fire
from tqdm import tqdm
from torchaudio.datasets import TEDLIUM
from scripts.eval.get_eval_set import get_eval_set
from scripts.eval.gen_inf_ckpt import gen_inf_ckpt

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

            self.dataset = Librispeech(
                root_dir=root_dir)
        elif eval_set == "librispeech_other":
            root_dir = f"{eval_dir}/librispeech_test_other"
            if not os.path.exists(root_dir):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

            self.dataset = Librispeech(
                root_dir=root_dir)
        elif eval_set == "artie_bias_corpus":
            root_dir = f"{eval_dir}/artie-bias-corpus"
            if not os.path.exists(root_dir):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

            self.dataset = ArtieBiasCorpus(
                root_dir=root_dir)
        elif eval_set == "fleurs":
            root_dir = f"{eval_dir}/fleurs"
            if not os.path.exists(root_dir):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)
                
            self.dataset = Fleurs(root_dir=root_dir)
        elif eval_set == "tedlium":
            if not os.path.exists(f"{eval_dir}/TEDLIUM_release-3"):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

            self.dataset = TEDLIUM(root=f"{eval_dir}", release="release3", subset="test")
        elif eval_set == "voxpopuli":
            root_dir = f"{eval_dir}/voxpopuli"
            if not os.path.exists(root_dir):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

            self.dataset = VoxPopuli(root_dir=root_dir)
        elif eval_set == "common_voice":
            if not os.path.exists(f"{eval_dir}/mozilla-foundation/common_voice_5_1"):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir, hf_token=hf_token)

            self.dataset = load_dataset(
                path="mozilla-foundation/common_voice_5_1",
                name="en",
                token=hf_token,
                split="test",)
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
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate, new_freq=16000
                )
                waveform = resampler(waveform)

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


def main(
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
    eval_dir: str = "data/eval",
    hf_token: Optional[str] = None,
):
    if "inf" not in ckpt:
        gen_inf_ckpt(ckpt, ckpt.replace(".pt", "_inf.pt"))
                
    device = torch.device("cuda")
    dataset = EvalDataset(eval_set=eval_set, hf_token=hf_token, eval_dir=eval_dir)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)

    model = load_model(name=ckpt, device=device, inference=True, in_memory=True)
    model.eval()

    normalizer = EnglishTextNormalizer()

    hypotheses = []
    references = []

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

        # avg_wer = total_wer / len(dataloader)
        avg_wer = jiwer.wer(references, hypotheses) * 100
        print(f"Average WER: {avg_wer}")


if __name__ == "__main__":
    Fire(main)
