from typing import Literal, Optional
import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
import jiwer
from whisper import audio, DecodingOptions
from whisper.normalizers import EnglishTextNormalizer, BasicTextNormalizer
from open_whisper import load_model
from fire import Fire
from tqdm import tqdm
from scripts.eval.get_eval_set import get_eval_set
from scripts.eval.gen_inf_ckpt import gen_inf_ckpt
import wandb
from tqdm import tqdm
from torchaudio.datasets import TEDLIUM
import torchaudio
from typing import Union
from pathlib import Path
from torchaudio.datasets.tedlium import _RELEASE_CONFIGS
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar

SHORT_FORM_EVAL_SETS = [
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

LONG_FORM_EVAL_SETS = ["tedlium_long"]


# short-form
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


class MLS:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load(self):
        main_dir = f"{self.root_dir}/test"
        transcript_fp = f"{main_dir}/transcripts.txt"
        audio_dir = f"{main_dir}/audio"

        with open(transcript_fp, "r") as f:
            audio_text_tpl = [line.strip().split("\t") for line in f]

        audio_text = {}
        for audio_file, text in audio_text_tpl:
            audio_fp = os.path.join(
                audio_dir,
                audio_file.split("_")[0],
                audio_file.split("_")[1],
                f"{audio_file}.opus",
            )
            audio_text[audio_fp] = text.strip()

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


# long-form
class TEDLIUM_long(TEDLIUM):
    def __init__(
        self,
        root: Union[str, Path],
        release: str = "release1",
        subset: str = "train",
        download: bool = False,
        audio_ext: str = ".sph",
    ) -> None:
        self._ext_audio = audio_ext
        if release in _RELEASE_CONFIGS.keys():
            folder_in_archive = _RELEASE_CONFIGS[release]["folder_in_archive"]
            url = _RELEASE_CONFIGS[release]["url"]
            subset = subset if subset else _RELEASE_CONFIGS[release]["subset"]
        else:
            # Raise warning
            raise RuntimeError(
                "The release {} does not match any of the supported tedlium releases{} ".format(
                    release,
                    _RELEASE_CONFIGS.keys(),
                )
            )
        if subset not in _RELEASE_CONFIGS[release]["supported_subsets"]:
            # Raise warning
            raise RuntimeError(
                "The subset {} does not match any of the supported tedlium subsets{} ".format(
                    subset,
                    _RELEASE_CONFIGS[release]["supported_subsets"],
                )
            )

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(".")[0]

        if release == "release3":
            if subset == "train":
                self._path = os.path.join(
                    root, folder_in_archive, _RELEASE_CONFIGS[release]["data_path"]
                )
            else:
                self._path = os.path.join(root, folder_in_archive, "legacy", subset)
        else:
            self._path = os.path.join(
                root, folder_in_archive, _RELEASE_CONFIGS[release]["data_path"], subset
            )

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _RELEASE_CONFIGS[release]["checksum"]
                    download_url_to_file(url, archive, hash_prefix=checksum)
                _extract_tar(archive)
        else:
            if not os.path.exists(self._path):
                raise RuntimeError(
                    f"The path {self._path} doesn't exist. "
                    "Please check the ``root`` path or set `download=True` to download it"
                )

        # Create list for all samples
        self._filelist = []
        stm_path = os.path.join(self._path, "stm")
        for file in sorted(os.listdir(stm_path)):
            if file.endswith(".stm"):
                # stm_path = os.path.join(self._path, "stm", file)
                # with open(stm_path) as f:
                #     l = len(f.readlines())
                #     file = file.replace(".stm", "")
                #     self._filelist.extend((file, line) for line in range(l))
                file = file.replace(".stm", "")
                self._filelist.append(file)
        # Create dict path for later read
        self._dict_path = os.path.join(
            root, folder_in_archive, _RELEASE_CONFIGS[release]["dict"]
        )
        self._phoneme_dict = None

    def _load_tedlium_item(self, fileid, path):
        transcript_path = os.path.join(path, "stm", fileid)
        transcript_segs = []
        init_start_time = None
        final_end_time = None
        init_talk_id = None
        init_speaker_id = None
        init_identifier = None

        with open(transcript_path + ".stm") as f:
            lines = f.readlines()
        l = len(lines)

        for i, line in enumerate(lines):
            talk_id, _, speaker_id, start_time, end_time, identifier, transcript_seg = (
                line.split(" ", 6)
            )

            if i == 0:
                init_start_time = start_time

            if i == 1:
                init_talk_id = talk_id
                init_speaker_id = speaker_id
                init_identifier = identifier

            if i == l - 1:
                final_end_time = end_time

            # transcript_segs.append(transcript_seg.strip())

            if transcript_seg.strip() != "ignore_time_segment_in_scoring":
                transcript_segs.append(transcript_seg.strip())
            # transcript = f.readlines()[line]
            # talk_id, _, speaker_id, start_time, end_time, identifier, transcript = transcript.split(" ", 6)

        start_time = init_start_time
        end_time = final_end_time
        talk_id = init_talk_id
        speaker_id = init_speaker_id
        identifier = init_identifier

        transcript = " ".join(transcript_segs)

        wave_path = os.path.join(path, "sph", fileid)
        waveform, sample_rate = self._load_audio(
            wave_path + self._ext_audio, start_time=start_time, end_time=end_time
        )

        return (waveform, sample_rate, transcript, talk_id, speaker_id, identifier)

    def _load_audio(self, path, start_time, end_time, sample_rate=16000):
        start_time = int(float(start_time) * sample_rate)
        end_time = int(float(end_time) * sample_rate)

        kwargs = {"frame_offset": start_time, "num_frames": end_time - start_time}

        return torchaudio.load(path, **kwargs)

    def __getitem__(self, n):
        fileid = self._filelist[n]
        return self._load_tedlium_item(fileid, self._path)


# multilingual
class EvalDataset(Dataset):
    def __init__(
        self,
        eval_set: Literal[
            "librispeech_clean",
            "librispeech_other",
            "multilingual_librispeech",
            "artie_bias_corpus",
            "fleurs",
            "tedlium",
            "tedlium_long",
            "voxpopuli",
            "common_voice",
            "ami_ihm",
            "ami_sdm",
        ],
        lang: Optional[str] = None,
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
        elif eval_set == "multilingual_librispeech":
            root_dir = f"{eval_dir}/mls/mls_{lang}_opus"
            if not os.path.exists(root_dir):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir, lang=lang)

            self.dataset = MLS(root_dir=root_dir)
        elif eval_set == "artie_bias_corpus":
            root_dir = f"{eval_dir}/artie-bias-corpus"
            if not os.path.exists(root_dir):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

            self.dataset = ArtieBiasCorpus(root_dir=root_dir)
        elif eval_set == "fleurs":
            if not os.path.exists(f"{eval_dir}/google___fleurs"):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

            self.dataset = load_dataset(
                path="google/fleurs",
                name="en_us",
                split="test",
                cache_dir=eval_dir,
                trust_remote_code=True,
                num_proc=15,
                save_infos=True,
            )
        elif eval_set == "tedlium":
            if not os.path.exists(f"{eval_dir}/TEDLIUM_release-3"):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

            self.dataset = TEDLIUM(
                root=f"{eval_dir}", release="release3", subset="test"
            )
        elif eval_set == "tedlium_long":
            eval_set = "tedlium"
            if not os.path.exists(f"{eval_dir}/TEDLIUM_release-3"):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

            self.dataset = TEDLIUM_long(
                root=f"{eval_dir}", release="release3", subset="test"
            )
        elif eval_set == "voxpopuli":
            if not os.path.exists(f"{eval_dir}/facebook___voxpopuli"):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

            self.dataset = load_dataset(
                path="facebook/voxpopuli",
                name="en",
                split="test",
                cache_dir=eval_dir,
                trust_remote_code=True,
                num_proc=15,
                save_infos=True,
            )
        elif eval_set == "common_voice":
            if not os.path.exists(f"{eval_dir}/mozilla-foundation___common_voice_5_1"):
                get_eval_set(eval_set=eval_set, eval_dir=eval_dir, hf_token=hf_token)

            self.dataset = load_dataset(
                path="mozilla-foundation/common_voice_5_1",
                name="en",
                split="test",
                token=hf_token,
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

        if self.eval_set not in [
            "tedlium",
            "tedlium_long",
            "common_voice",
            "fleurs",
            "voxpopuli",
        ]:
            audio_files, transcript_texts = self.dataset.load()
            self.audio_files = audio_files
            self.transcript_texts = transcript_texts

    def __len__(self):
        if self.eval_set in [
            "tedlium",
            "tedlium_long",
            "common_voice",
            "fleurs",
            "voxpopuli",
        ]:
            return len(self.dataset)
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_fp = ""
        audio_arr = ""

        if self.eval_set == "tedlium":
            waveform, _, text_y, *_ = self.dataset[index]
            audio_arr = audio.pad_or_trim(waveform[0])
            audio_input = audio.log_mel_spectrogram(audio_arr)
        elif self.eval_set == "tedlium_long":
            audio_arr, _, text_y, *_ = self.dataset[index]
            audio_input = None
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
        elif self.eval_set == "fleurs":
            waveform = self.dataset[index]["audio"]["array"]
            sampling_rate = self.dataset[index]["audio"]["sampling_rate"]
            text_y = self.dataset[index]["transcription"]

            if sampling_rate != 16000:
                waveform = librosa.resample(
                    waveform, orig_sr=sampling_rate, target_sr=16000
                )

            audio_arr = audio.pad_or_trim(waveform)
            audio_arr = audio_arr.astype(np.float32)
            audio_input = audio.log_mel_spectrogram(audio_arr)
        elif self.eval_set == "voxpopuli":
            waveform = self.dataset[index]["audio"]["array"]
            sampling_rate = self.dataset[index]["audio"]["sampling_rate"]
            text_y = self.dataset[index]["normalized_text"]

            if sampling_rate != 16000:
                waveform = librosa.resample(
                    waveform, orig_sr=sampling_rate, target_sr=16000
                )

            audio_arr = audio.pad_or_trim(waveform)
            audio_arr = audio_arr.astype(np.float32)
            audio_input = audio.log_mel_spectrogram(audio_arr)
        else:
            audio_fp = self.audio_files[index]
            audio_arr, audio_input = self.preprocess_audio(audio_fp)
            text_y = self.transcript_texts[index]

        return audio_fp, audio_arr, audio_input, text_y

    def preprocess_audio(self, audio_file):
        audio_arr = audio.load_audio(audio_file, sr=16000)
        audio_arr = audio.pad_or_trim(audio_arr)
        mel_spec = audio.log_mel_spectrogram(audio_arr)
        return audio_arr, mel_spec


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
    log_dir: str,
    current_step: Optional[int] = None,
    exp_name: Optional[str] = None,
    wandb_log: bool = False,
    wandb_run_id: Optional[str] = None,
    wandb_log_dir: str = "wandb",
    eval_dir: str = "data/eval",
    hf_token: Optional[str] = None,
):
    if "inf" not in ckpt and ckpt.split("/")[-2] != "whisper_ckpts":
        ckpt = gen_inf_ckpt(ckpt, ckpt.replace(".pt", "_inf.pt"))

    os.makedirs(log_dir, exist_ok=True)
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
        wandb_table_cols = [
            "eval_set",
            "audio",
            "prediction",
            "target",
            "subs",
            "dels",
            "ins",
            "wer",
        ]
        if wandb_run_id:
            wandb_table_cols.append("run_id")
            wandb.init(
                id=wandb_run_id,
                resume="allow",
                project="open_whisper",
                entity="dogml",
                save_code=True,
                settings=wandb.Settings(init_timeout=300, _service_wait=300),
            )
        else:
            run_id = wandb.util.generate_id()
            ow_or_w = "open-whisper" if ckpt.split("/")[-3] == "ow_ckpts" else "whisper"
            exp_name = (
                f"{eval_set}_eval" if ow_or_w == "whisper" else f"ow_{eval_set}_eval"
            )
            model_sizes = ["tiny", "small", "base", "medium", "large"]
            model_size = [
                model_size for model_size in model_sizes if model_size in ckpt
            ][0]
            config = {
                "ckpt": "/".join(ckpt.split("/")[-2:]),
                "model": ow_or_w,
                "model_size": model_size,
            }
            wandb.init(
                id=run_id,
                resume="allow",
                project="open_whisper",
                entity="dogml",
                job_type="evals",
                name=exp_name,
                dir=wandb_log_dir,
                config=config,
                tags=["eval", eval_set, ow_or_w, model_size],
            )
        eval_table = wandb.Table(columns=wandb_table_cols)
        table_iter = 0

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            _, audio_arr, audio_input, text_y = batch

            norm_tgt_text = [normalizer(text) for text in text_y]
            audio_input = audio_input.to(device)

            options = DecodingOptions(language="en", without_timestamps=True)

            results = model.decode(audio_input, options=options)

            norm_pred_text = [
                normalizer(results[i].text)
                for i in range(len(results))
                if norm_tgt_text[i] != ""
                and norm_tgt_text[i] != "ignore time segment in scoring"
            ]
            if wandb_log:
                audio_arr = [
                    audio_arr.numpy()[i]
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
                table_iter += 1
                for i in tqdm(range(0, len(norm_pred_text)), total=len(norm_pred_text)):
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

                    if wandb_run_id:
                        eval_table.add_data(
                            eval_set,
                            wandb.Audio(audio_arr[i], sample_rate=16000),
                            norm_pred_text[i],
                            norm_tgt_text[i],
                            subs,
                            dels,
                            ins,
                            wer,
                            wandb_run_id,
                        )
                    else:
                        eval_table.add_data(
                            eval_set,
                            wandb.Audio(audio_arr[i], sample_rate=16000),
                            norm_pred_text[i],
                            norm_tgt_text[i],
                            subs,
                            dels,
                            ins,
                            wer,
                        )

                wandb.log({f"eval_table_{table_iter}": eval_table})
                eval_table = wandb.Table(columns=wandb_table_cols)

    avg_wer = jiwer.wer(references, hypotheses) * 100
    avg_measures = jiwer.compute_measures(truth=references, hypothesis=hypotheses)
    avg_subs = avg_measures["substitutions"]
    avg_ins = avg_measures["insertions"]
    avg_dels = avg_measures["deletions"]

    print(
        f"Average WER: {avg_wer}, Average Subs: {avg_subs}, Average Ins: {avg_ins}, Average Dels: {avg_dels}"
    )

    if wandb_log:
        if wandb_run_id:
            wandb.log({f"eval/{eval_set}_wer": avg_wer, "custom_step": current_step})
            wandb.log({f"eval/{eval_set}_subs": avg_subs, "custom_step": current_step})
            wandb.log({f"eval/{eval_set}_ins": avg_ins, "custom_step": current_step})
            wandb.log({f"eval/{eval_set}_dels": avg_dels, "custom_step": current_step})
        else:
            wandb.run.summary["avg_wer"] = avg_wer
            wandb.run.summary["avg_subs"] = avg_subs
            wandb.run.summary["avg_ins"] = avg_ins
            wandb.run.summary["avg_dels"] = avg_dels

            with open(
                f"{log_dir}/training/{exp_name}/{wandb_run_id}/eval_results.txt", "a"
            ) as f:
                f.write(
                    f"{eval_set} WER: {avg_wer}, Subs: {avg_subs}, Ins: {avg_ins}, Dels: {avg_dels}\n"
                )



def long_form_eval(
    batch_size: int,
    num_workers: int,
    ckpt: str,
    eval_set: Literal["tedlium_long",],
    log_dir: str,
    current_step: Optional[int] = None,
    exp_name: Optional[str] = None,
    wandb_log: bool = False,
    wandb_run_id: Optional[str] = None,
    wandb_log_dir: str = "wandb",
    eval_dir: str = "data/eval",
    hf_token: Optional[str] = None,
) -> None:
    if "inf" not in ckpt and ckpt.split("/")[-2] != "whisper_ckpts":
        ckpt = gen_inf_ckpt(ckpt, ckpt.replace(".pt", "_inf.pt"))

    os.makedirs(log_dir, exist_ok=True)
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
        wandb_table_cols = [
            "eval_set",
            "audio",
            "prediction",
            "target",
            "subs",
            "dels",
            "ins",
            "wer",
        ]
        if wandb_run_id:
            wandb_table_cols.append("run_id")
            wandb.init(
                id=wandb_run_id,
                resume="allow",
                project="open_whisper",
                entity="dogml",
                save_code=True,
                settings=wandb.Settings(init_timeout=300, _service_wait=300),
            )
        else:
            run_id = wandb.util.generate_id()
            ow_or_w = "open-whisper" if ckpt.split("/")[-3] == "ow_ckpts" else "whisper"
            exp_name = (
                f"{eval_set}_eval" if ow_or_w == "whisper" else f"ow_{eval_set}_eval"
            )
            model_sizes = ["tiny", "small", "base", "medium", "large"]
            model_size = [
                model_size for model_size in model_sizes if model_size in ckpt
            ][0]
            config = {
                "ckpt": "/".join(ckpt.split("/")[-2:]),
                "model": ow_or_w,
                "model_size": model_size,
            }
            wandb.init(
                id=run_id,
                resume="allow",
                project="open_whisper",
                entity="dogml",
                job_type="evals",
                name=exp_name,
                dir=wandb_log_dir,
                config=config,
                tags=["eval", eval_set, ow_or_w, model_size],
            )
        eval_table = wandb.Table(columns=wandb_table_cols)
        table_iter = 0

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            _, audio_input, _, text_y = batch

            norm_tgt_text = [normalizer(text) for text in text_y]
            audio_input = audio_input.to(device)

            options = dict(
                task="transcribe",
                language="en",
                without_timestamps=False,
                beam_size=5,
                best_of=5,
            )
            results = model.transcribe(audio_input[0], **options)

            norm_pred_text = [
                normalizer(results["text"])
                for i in range(len(norm_tgt_text))
                if norm_tgt_text[i] != ""
            ]
            if wandb_log:
                audio_arr = [
                    audio_arr.numpy()[i]
                    for i in range(len(norm_tgt_text))
                    if norm_tgt_text[i] != ""
                ]
            norm_tgt_text = [
                norm_tgt_text[i]
                for i in range(len(norm_tgt_text))
                if norm_tgt_text[i] != ""
            ]

            references.extend(norm_tgt_text)
            hypotheses.extend(norm_pred_text)

    avg_wer = jiwer.wer(references, hypotheses) * 100
    avg_measures = jiwer.compute_measures(truth=references, hypothesis=hypotheses)
    avg_subs = avg_measures["substitutions"]
    avg_ins = avg_measures["insertions"]
    avg_dels = avg_measures["deletions"]

    if wandb_log:
        if wandb_run_id:
            wandb.log(
                {
                    f"eval/{eval_set}_{lang}_wer": avg_wer,
                    "custom_step": current_step,
                }
            )
            wandb.log(
                {
                    f"eval/{eval_set}_{lang}_subs": avg_subs,
                    "custom_step": current_step,
                }
            )
            wandb.log(
                {
                    f"eval/{eval_set}_{lang}_ins": avg_ins,
                    "custom_step": current_step,
                }
            )
            wandb.log(
                {
                    f"eval/{eval_set}_{lang}_dels": avg_dels,
                    "custom_step": current_step,
                }
            )
        else:
            wandb.run.summary[f"avg_{lang}_wer"] = avg_wer
            wandb.run.summary[f"avg_{lang}_subs"] = avg_subs
            wandb.run.summary[f"avg_{lang}_ins"] = avg_ins
            wandb.run.summary[f"avg_{lang}_dels"] = avg_dels
    else:
        if exp_name is not None and wandb_run_id is not None:
            path = f"{log_dir}/training/{exp_name}/{wandb_run_id}/eval_results.txt"
        else:
            path = f"{log_dir}/eval_results.txt"
        with open(path, "a") as f:
            f.write(
                f"{eval_set} {lang} WER: {avg_wer}, Subs: {avg_subs}, Ins: {avg_ins}, Dels: {avg_dels}\n"
            )

    print(
        f"Language: {lang}, Average WER: {avg_wer}, Average Subs: {avg_subs}, Average Ins: {avg_ins}, Average Dels: {avg_dels}"
    )


def ml_eval(
    batch_size: int,
    num_workers: int,
    ckpt: str,
    eval_set: Literal["multilingual_librispeech",],
    log_dir: str,
    lang: Optional[str] = None,
    current_step: Optional[int] = None,
    exp_name: Optional[str] = None,
    wandb_log: bool = False,
    wandb_run_id: Optional[str] = None,
    wandb_log_dir: str = "wandb",
    eval_dir: str = "data/eval",
    hf_token: Optional[str] = None,
):
    if "inf" not in ckpt and ckpt.split("/")[-2] != "whisper_ckpts":
        ckpt = gen_inf_ckpt(ckpt, ckpt.replace(".pt", "_inf.pt"))

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(wandb_log_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device("cuda")

    model = load_model(name=ckpt, device=device, inference=True, in_memory=True)
    model.eval()

    normalizer = BasicTextNormalizer()

    if lang is None:
        if eval_set == "multilingual_librispeech":
            langs = [
                "german",
                "dutch",
                "spanish",
                "french",
                "italian",
                "portuguese",
                "polish",
            ]
    else:
        langs = [lang]

    if wandb_log:
        wandb_table_cols = [
            "eval_set",
            "lang",
            "audio",
            "prediction",
            "target",
            "subs",
            "dels",
            "ins",
            "wer",
        ]
        if wandb_run_id:
            wandb_table_cols.append("run_id")
            wandb.init(
                id=wandb_run_id,
                resume="allow",
                project="open_whisper",
                entity="dogml",
                save_code=True,
                settings=wandb.Settings(init_timeout=300, _service_wait=300),
            )
        else:
            run_id = wandb.util.generate_id()
            ow_or_w = "open-whisper" if ckpt.split("/")[-3] == "ow_ckpts" else "whisper"
            exp_name = (
                f"{eval_set}_eval" if ow_or_w == "whisper" else f"ow_{eval_set}_eval"
            )
            model_sizes = ["tiny", "small", "base", "medium", "large"]
            model_size = [
                model_size for model_size in model_sizes if model_size in ckpt
            ][0]
            config = {
                "ckpt": "/".join(ckpt.split("/")[-2:]),
                "model": ow_or_w,
                "model_size": model_size,
            }
            wandb.init(
                id=run_id,
                resume="allow",
                project="open_whisper",
                entity="dogml",
                job_type="evals",
                name=exp_name,
                dir=wandb_log_dir,
                config=config,
                tags=[
                    "eval",
                    "multilingual",
                    "all_langs" if lang is None else lang,
                    eval_set,
                    ow_or_w,
                    model_size,
                ],
            )
        eval_table = wandb.Table(columns=wandb_table_cols)
        table_iter = 0

    for lang in langs:
        dataset = EvalDataset(
            eval_set=eval_set, lang=lang, hf_token=hf_token, eval_dir=eval_dir
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )

        hypotheses = []
        references = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                _, audio_input, mel_spec, text_y = batch

                norm_tgt_text = [normalizer(text) for text in text_y]
                audio_input = audio_input.to(device)

                options = dict(
                    task="transcribe",
                    language=lang,
                    without_timestamps=True,
                    beam_size=5,
                    best_of=5,
                )
                results = model.transcribe(audio_input[0], **options)

                norm_pred_text = [
                    normalizer(results["text"])
                    for i in range(len(norm_tgt_text))
                    if norm_tgt_text[i] != ""
                ]
                if wandb_log:
                    audio_arr = [
                        audio_arr.numpy()[i]
                        for i in range(len(norm_tgt_text))
                        if norm_tgt_text[i] != ""
                    ]
                norm_tgt_text = [
                    norm_tgt_text[i]
                    for i in range(len(norm_tgt_text))
                    if norm_tgt_text[i] != ""
                ]

                references.extend(norm_tgt_text)
                hypotheses.extend(norm_pred_text)

        avg_wer = jiwer.wer(references, hypotheses) * 100
        avg_measures = jiwer.compute_measures(truth=references, hypothesis=hypotheses)
        avg_subs = avg_measures["substitutions"]
        avg_ins = avg_measures["insertions"]
        avg_dels = avg_measures["deletions"]

        if wandb_log:
            if wandb_run_id:
                wandb.log(
                    {
                        f"eval/{eval_set}_{lang}_wer": avg_wer,
                        "custom_step": current_step,
                    }
                )
                wandb.log(
                    {
                        f"eval/{eval_set}_{lang}_subs": avg_subs,
                        "custom_step": current_step,
                    }
                )
                wandb.log(
                    {
                        f"eval/{eval_set}_{lang}_ins": avg_ins,
                        "custom_step": current_step,
                    }
                )
                wandb.log(
                    {
                        f"eval/{eval_set}_{lang}_dels": avg_dels,
                        "custom_step": current_step,
                    }
                )
            else:
                wandb.run.summary[f"avg_{lang}_wer"] = avg_wer
                wandb.run.summary[f"avg_{lang}_subs"] = avg_subs
                wandb.run.summary[f"avg_{lang}_ins"] = avg_ins
                wandb.run.summary[f"avg_{lang}_dels"] = avg_dels
        else:
            if exp_name is not None and wandb_run_id is not None:
                path = f"{log_dir}/training/{exp_name}/{wandb_run_id}/eval_results.txt"
            else:
                path = f"{log_dir}/eval_results.txt"
            with open(path, "a") as f:
                f.write(
                    f"{eval_set} {lang} WER: {avg_wer}, Subs: {avg_subs}, Ins: {avg_ins}, Dels: {avg_dels}\n"
                )

        print(
            f"Language: {lang}, Average WER: {avg_wer}, Average Subs: {avg_subs}, Average Ins: {avg_ins}, Average Dels: {avg_dels}"
        )


if __name__ == "__main__":
    Fire({"main": main, "ml_eval": ml_eval, "long_form_eval": long_form_eval})
