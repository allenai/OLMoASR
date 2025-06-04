from typing import Literal, Optional
import os
import glob
import json
import re
import io
import numpy as np
import pandas as pd
from collections import OrderedDict
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.amp import autocast
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
)
from nemo.collections.asr.models import ASRModel
import jiwer
from whisper import audio, DecodingOptions
from whisper.tokenizer import get_tokenizer
from whisper.decoding import detect_language
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
import csv
import subprocess

# voxlingua imports
# from speechbrain.inference.classifiers import EncoderClassifier

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
ML_EVAL_SETS = ["fleurs"]
TRANSLATE_EVAL_SETS = []
LANG_ID_EVAL_SETS = ["fleurs"]

FLEURS_LANG_TO_ID = OrderedDict(
    [
        ("Afrikaans", "af"),
        ("Amharic", "am"),
        ("Arabic", "ar"),
        ("Armenian", "hy"),
        ("Assamese", "as"),
        ("Asturian", "ast"),
        ("Azerbaijani", "az"),
        ("Belarusian", "be"),
        ("Bengali", "bn"),
        ("Bosnian", "bs"),
        ("Bulgarian", "bg"),
        ("Burmese", "my"),
        ("Catalan", "ca"),
        ("Cebuano", "ceb"),
        ("Mandarin Chinese", "cmn_hans"),
        ("Cantonese Chinese", "yue_hant"),
        ("Croatian", "hr"),
        ("Czech", "cs"),
        ("Danish", "da"),
        ("Dutch", "nl"),
        ("English", "en"),
        ("Estonian", "et"),
        ("Filipino", "fil"),
        ("Finnish", "fi"),
        ("French", "fr"),
        ("Fula", "ff"),
        ("Galician", "gl"),
        ("Ganda", "lg"),
        ("Georgian", "ka"),
        ("German", "de"),
        ("Greek", "el"),
        ("Gujarati", "gu"),
        ("Hausa", "ha"),
        ("Hebrew", "he"),
        ("Hindi", "hi"),
        ("Hungarian", "hu"),
        ("Icelandic", "is"),
        ("Igbo", "ig"),
        ("Indonesian", "id"),
        ("Irish", "ga"),
        ("Italian", "it"),
        ("Japanese", "ja"),
        ("Javanese", "jv"),
        ("Kabuverdianu", "kea"),
        ("Kamba", "kam"),
        ("Kannada", "kn"),
        ("Kazakh", "kk"),
        ("Khmer", "km"),
        ("Korean", "ko"),
        ("Kyrgyz", "ky"),
        ("Lao", "lo"),
        ("Latvian", "lv"),
        ("Lingala", "ln"),
        ("Lithuanian", "lt"),
        ("Luo", "luo"),
        ("Luxembourgish", "lb"),
        ("Macedonian", "mk"),
        ("Malay", "ms"),
        ("Malayalam", "ml"),
        ("Maltese", "mt"),
        ("Maori", "mi"),
        ("Marathi", "mr"),
        ("Mongolian", "mn"),
        ("Nepali", "ne"),
        ("Northern-Sotho", "nso"),
        ("Norwegian", "nb"),
        ("Nyanja", "ny"),
        ("Occitan", "oc"),
        ("Oriya", "or"),
        ("Oromo", "om"),
        ("Pashto", "ps"),
        ("Persian", "fa"),
        ("Polish", "pl"),
        ("Portuguese", "pt"),
        ("Punjabi", "pa"),
        ("Romanian", "ro"),
        ("Russian", "ru"),
        ("Serbian", "sr"),
        ("Shona", "sn"),
        ("Sindhi", "sd"),
        ("Slovak", "sk"),
        ("Slovenian", "sl"),
        ("Somali", "so"),
        ("Sorani-Kurdish", "ckb"),
        ("Spanish", "es"),
        ("Swahili", "sw"),
        ("Swedish", "sv"),
        ("Tajik", "tg"),
        ("Tamil", "ta"),
        ("Telugu", "te"),
        ("Thai", "th"),
        ("Turkish", "tr"),
        ("Ukrainian", "uk"),
        ("Umbundu", "umb"),
        ("Urdu", "ur"),
        ("Uzbek", "uz"),
        ("Vietnamese", "vi"),
        ("Welsh", "cy"),
        ("Wolof", "wo"),
        ("Xhosa", "xh"),
        ("Yoruba", "yo"),
        ("Zulu", "zu"),
    ]
)

OW_TO_FLEURS = {
    "no": "nb",
    "jw": "jv",
    "zh": "cmn_hans",
    "yue": "yue_hant",
    "tl": "fil",
}
OW_NOT_IN_FLEURS = [
    "la",
    "br",
    "eu",
    "sq",
    "si",
    "yi",
    "fo",
    "ht",
    "tk",
    "nn",
    "sa",
    "bo",
    "tl",
    "mg",
    "tt",
    "haw",
    "ba",
    "su",
    "ht",
    "si",
]
FLEURS_NOT_IN_OW = [
    "ast",
    "ceb",
    "ff",
    "lg",
    "ig",
    "ga",
    "kea",
    "kam",
    "ky",
    "luo",
    "nso",
    "ny",
    "or",
    "om",
    "ckb",
    "umb",
    "wo",
    "xh",
    "zu",
]

# for voxlingua107 lang id model
VOX_TO_FLEURS = {"iw": "he", "zh": "cmn_hans", "jw": "jv", "no": "nb", "tl": "fil"}
VOX_NOT_IN_FLEURS = [
    "ab",
    "ba",
    "bo",
    "br",
    "eo",
    "eu",
    "fo",
    "gn",
    "gv",
    "haw",
    "ht",
    "ia",
    "la",
    "mg",
    "nn",
    "sa",
    "sco",
    "si",
    "sq",
    "su",
    "tk",
    "tt",
    "war",
    "yi",
]
FLEURS_NOT_IN_VOX = [
    "ast",
    "ff",
    "lg",
    "ig",
    "ga",
    "kea",
    "kam",
    "ky",
    "luo",
    "nso",
    "ny",
    "or",
    "om",
    "ckb",
    "umb",
    "wo",
    "xh",
    "zu",
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


class CORAAL:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load(self):
        audio_files = []
        transcript_texts = []

        segments = pd.read_csv(
            f"{self.root_dir}/CORAAL_transcripts.csv", quotechar='"'
        ).values.tolist()

        def remove_markers(line, markers):
            # Remove any text within markers, e.g. 'We(BR) went' -> 'We went'
            # markers = list of pairs, e.g. ['()', '[]'] denoting breath or noise in transcripts
            for s, e in markers:
                line = re.sub(" ?\\" + s + "[^" + e + "]+\\" + e, "", line)
            return line

        def clean_within_coraal(text):

            text = text.replace("\[", "\{")
            text = text.replace("\]", "\}")
            # Relabel CORAAL words. For consideration: aks -> ask?
            split_words = text.split()
            split_words = [x if x != "busses" else "buses" for x in split_words]
            split_words = [x if x != "aks" else "ask" for x in split_words]
            split_words = [x if x != "aksing" else "asking" for x in split_words]
            split_words = [x if x != "aksed" else "asked" for x in split_words]
            text = " ".join(split_words)

            # remove CORAAL unintelligible flags
            text = re.sub("(?i)\/unintelligible\/", "", "".join(text))
            text = re.sub("(?i)\/inaudible\/", "", "".join(text))
            text = re.sub("\/RD(.*?)\/", "", "".join(text))
            text = re.sub("\/(\?)\1*\/", "", "".join(text))

            # remove nonlinguistic markers
            text = remove_markers(text, ["<>", "()", "{}"])

            return text

        for segment in segments:
            segment_filename, _, _, _, source, _, _, content, *_ = segment
            sub_folder = os.path.join(self.root_dir, "CORAAL_audio", source.lower())
            audio_file = os.path.join(sub_folder, segment_filename)
            if not os.path.exists(audio_file):
                audio_file = audio_file.replace(".wav", ".mp3")
            audio_files.append(audio_file)
            content = clean_within_coraal(content)
            transcript_texts.append(content)

        return audio_files, transcript_texts


class chime6:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load(self):
        audio_files = []
        transcript_texts = []

        for p in glob.glob(f"{self.root_dir}/transcripts/*.json"):
            with open(p, "r") as f:
                data = json.load(f)
                audio_files.extend(
                    [
                        os.path.join(
                            self.root_dir, "segments", segment_dict["audio_seg_file"]
                        )
                        for segment_dict in data
                    ]
                )
                transcript_texts.extend(
                    [segment_dict["words"] for segment_dict in data]
                )

        return audio_files, transcript_texts


class WSJ:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load(self):
        audio_files = []
        transcript_files = []

        for direc in glob.glob(f"{self.root_dir}/test_eval*"):
            id_2_text = {}
            with open(f"{direc}/text") as f:
                id_2_text = {
                    line.strip()
                    .split(" ")[0]: line.strip()
                    .split(" ", maxsplit=1)[-1]
                    .strip()
                    for line in f
                }

            with open(f"{direc}/wav.scp") as f:
                for line in f:
                    audio_file = line.strip().split(" ", maxsplit=1)[-1].split(" |")[0]
                    utter_id = line.strip().split(" ")[0]
                    transcript_text = id_2_text[utter_id]
                    audio_files.append(audio_file)
                    transcript_files.append(transcript_text)

        return audio_files, transcript_files


class CallHome:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load(self):
        audio_files = []
        transcript_files = []

        with open(
            f"{self.root_dir}/2000_hub5_eng_eval_tr/reference/hub5e00.english.000405.stm",
            "r",
        ) as f:
            for line in f:
                if line.startswith(";;"):
                    continue
                elif line.startswith("en"):
                    audio_file = (
                        f"{self.root_dir}/hub5e_00/english/"
                        + line.split(" ")[0]
                        + ".sph"
                    )
                    channel = line.split(" ")[1]
                    if channel == "A":
                        wav_file = audio_file.split(".")[0] + "_A.wav"
                        if not os.path.exists(wav_file):
                            _ = subprocess.run(
                                ["sox", audio_file, wav_file, "remix", "1"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                            )
                    elif channel == "B":
                        wav_file = audio_file.split(".")[0] + "_B.wav"
                        if not os.path.exists(wav_file):
                            _ = subprocess.run(
                                ["sox", audio_file, wav_file, "remix", "2"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                            )
                    transcript_text = re.split(r"<[^>]+>", line)[-1].strip()
                    start_time = float(line.split(" ")[3])
                    if line.split(" ")[4] != "":
                        end_time = float(line.split(" ")[4])
                    elif line.split(" ")[5] != "":
                        end_time = float(line.split(" ")[5])
                    else:
                        end_time = float(line.split(" ")[6])
                    audio_files.append((wav_file, start_time, end_time))
                    transcript_files.append(transcript_text)

        return audio_files, transcript_files


class SwitchBoard:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load(self):
        audio_files = []
        transcript_files = []

        with open(
            f"{self.root_dir}/2000_hub5_eng_eval_tr/reference/hub5e00.english.000405.stm",
            "r",
        ) as f:
            for line in f:
                if line.startswith(";;"):
                    continue
                elif line.startswith("sw"):
                    audio_file = (
                        f"{self.root_dir}/hub5e_00/english/"
                        + line.split(" ")[0]
                        + ".sph"
                    )
                    channel = line.split(" ")[1]
                    if channel == "A":
                        wav_file = audio_file.split(".")[0] + "_A.wav"
                        if not os.path.exists(wav_file):
                            _ = subprocess.run(
                                ["sox", audio_file, wav_file, "remix", "1"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                            )
                    elif channel == "B":
                        wav_file = audio_file.split(".")[0] + "_B.wav"
                        if not os.path.exists(wav_file):
                            _ = subprocess.run(
                                ["sox", audio_file, wav_file, "remix", "2"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                            )
                    transcript_text = re.split(r"<[^>]+>", line)[-1].strip()
                    start_time = float(line.split(" ")[3])
                    if line.split(" ")[4] != "":
                        end_time = float(line.split(" ")[4])
                    elif line.split(" ")[5] != "":
                        end_time = float(line.split(" ")[5])
                    else:
                        end_time = float(line.split(" ")[6])
                    audio_files.append((wav_file, start_time, end_time))
                    transcript_files.append(transcript_text)

        return audio_files, transcript_files


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


# Kincaid46
class Kincaid46:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load(self):
        audio_files = []
        transcript_texts = []

        with open(f"{self.root_dir}/text.csv", "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                audio_file = os.path.join(self.root_dir, "audio", f"{(i - 1):02}.m4a")
                transcript_text = row[5]
                audio_files.append(audio_file)
                transcript_texts.append(transcript_text)

        return audio_files, transcript_texts


# CORAAL_long
class CORAAL_long:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load(self):
        audio_files = []
        transcript_texts = []

        with open(f"{self.root_dir}/coraal_transcripts.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                audio_files.append(data["audio"])
                transcript_texts.append(data["text"])

        return audio_files, transcript_texts


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


class EvalDataset(Dataset):
    def __init__(
        self,
        task: Literal[
            "eng_transcribe",
            "long_form_transcribe",
            "ml_transcribe",
            "translate",
            "lang_id",
        ],
        eval_set: Literal[
            "librispeech_clean",
            "librispeech_other",
            "multilingual_librispeech",
            "artie_bias_corpus",
            "fleurs",
            "tedlium",
            "voxpopuli",
            "common_voice",
            "ami_ihm",
            "ami_sdm",
        ],
        lang: Optional[str] = None,
        hf_token: Optional[str] = None,
        eval_dir: str = "data/eval",
        n_mels: int = 80,
    ):
        if task == "eng_transcribe":
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
                if not os.path.exists(f"{eval_dir}/google___fleurs/en_us"):
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
                if not os.path.exists(
                    f"{eval_dir}/mozilla-foundation___common_voice_5_1"
                ):
                    get_eval_set(
                        eval_set=eval_set, eval_dir=eval_dir, hf_token=hf_token
                    )

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
            elif eval_set == "coraal":
                root_dir = f"{eval_dir}/coraal"
                if not os.path.exists(root_dir):
                    get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

                self.dataset = CORAAL(root_dir=root_dir)
            elif eval_set == "chime6":
                root_dir = f"{eval_dir}/chime6"
                if not os.path.exists(root_dir):
                    get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

                self.dataset = chime6(root_dir=root_dir)
            elif eval_set == "wsj":
                root_dir = f"{eval_dir}/kaldi/egs/wsj/s5/data"
                self.dataset = WSJ(root_dir=root_dir)
            elif eval_set == "callhome":
                root_dir = eval_dir
                self.dataset = CallHome(root_dir=root_dir)
            elif eval_set == "switchboard":
                root_dir = eval_dir
                self.dataset = SwitchBoard(root_dir=root_dir)
        elif task == "long_form_transcribe":
            if eval_set == "tedlium":
                self.dataset = load_dataset(
                    path="distil-whisper/tedlium-long-form",
                    split="test",
                    token=hf_token,
                    cache_dir=eval_dir,
                    trust_remote_code=True,
                    num_proc=15,
                    save_infos=True,
                )

                # if not os.path.exists(f"{eval_dir}/TEDLIUM_release-3"):
                #     get_eval_set(eval_set=eval_set, eval_dir=eval_dir)

                # self.dataset = TEDLIUM_long(
                #     root=f"{eval_dir}", release="release3", subset="test"
                # )
            elif eval_set == "meanwhile":
                self.dataset = load_dataset(
                    path="distil-whisper/meanwhile",
                    split="test",
                    token=hf_token,
                    cache_dir=eval_dir,
                    trust_remote_code=True,
                    num_proc=15,
                    save_infos=True,
                )
            elif eval_set == "rev16":
                self.dataset = load_dataset(
                    path="distil-whisper/rev16",
                    name="whisper_subset",
                    split="test",
                    token=hf_token,
                    cache_dir=eval_dir,
                    trust_remote_code=True,
                    num_proc=15,
                    save_infos=True,
                )
            elif eval_set == "earnings21":
                self.dataset = load_dataset(
                    path="distil-whisper/earnings21",
                    name="full",
                    split="test",
                    token=hf_token,
                    cache_dir=eval_dir,
                    trust_remote_code=True,
                    num_proc=15,
                    save_infos=True,
                )
            elif eval_set == "earnings22":
                self.dataset = load_dataset(
                    path="distil-whisper/earnings22",
                    name="full",
                    split="test",
                    token=hf_token,
                    cache_dir=eval_dir,
                    trust_remote_code=True,
                    num_proc=15,
                    save_infos=True,
                )
            elif eval_set == "coraal":
                if not os.path.exists(f"{eval_dir}/coraal_long"):
                    get_eval_set(eval_set=eval_set, eval_dir=eval_dir)
                root_dir = f"{eval_dir}/coraal_long"
                self.dataset = CORAAL_long(root_dir=root_dir)
            elif eval_set == "kincaid46":
                if not os.path.exists(f"{eval_dir}/kincaid46"):
                    get_eval_set(eval_set=eval_set, eval_dir=eval_dir)
                root_dir = f"{eval_dir}/kincaid46"
                self.dataset = Kincaid46(root_dir=root_dir)
        elif task == "ml_transcribe":
            if eval_set == "fleurs":
                if len(os.listdir(f"{eval_dir}/google__fleurs")) < 102:
                    get_eval_set(eval_set="fleurs", eval_dir=eval_dir)

                self.dataset = load_dataset(
                    path="google/fleurs",
                    name="all",
                    split="test",
                    cache_dir=eval_dir,
                    trust_remote_code=True,
                    num_proc=15,
                    save_infos=True,
                )
            else:
                if not os.path.exists(f"{eval_dir}/google___fleurs/{lang}"):
                    get_eval_set(eval_set="fleurs", eval_dir=eval_dir, lang=lang)

                self.dataset = load_dataset(
                    path="google/fleurs",
                    name=lang,
                    split="test",
                    cache_dir=eval_dir,
                    trust_remote_code=True,
                    num_proc=15,
                    save_infos=True,
                )
        elif task == "translate":
            pass
        elif task == "lang_id":
            if not os.path.exists(f"{eval_dir}/google___fleurs/all"):
                get_eval_set(eval_set="fleurs", lang="all", eval_dir=eval_dir)

            self.dataset = load_dataset(
                path="google/fleurs",
                name="all",
                split="test",
                cache_dir=eval_dir,
                trust_remote_code=True,
                num_proc=15,
                save_infos=True,
            )

        self.eval_set = eval_set
        self.task = task
        self.n_mels = n_mels

        if self.eval_set not in [
            "tedlium",
            "common_voice",
            "fleurs",
            "voxpopuli",
            "meanwhile",
            "rev16",
            "earnings21",
            "earnings22",
        ]:
            audio_files, transcript_texts = self.dataset.load()
            self.audio_files = audio_files
            self.transcript_texts = transcript_texts

    def __len__(self):
        if self.eval_set in [
            "tedlium",
            "common_voice",
            "fleurs",
            "voxpopuli",
            "meanwhile",
            "rev16",
            "earnings21",
            "earnings22",
        ]:
            return len(self.dataset)
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_fp = ""
        audio_arr = ""

        if self.task == "eng_transcribe":
            if self.eval_set == "tedlium":
                waveform, _, text_y, *_ = self.dataset[index]
                audio_arr = audio.pad_or_trim(waveform[0])
                audio_input = audio.log_mel_spectrogram(audio_arr, n_mels=self.n_mels)
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
                audio_input = audio.log_mel_spectrogram(audio_arr, n_mels=self.n_mels)
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
                audio_input = audio.log_mel_spectrogram(audio_arr, n_mels=self.n_mels)
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
                audio_input = audio.log_mel_spectrogram(audio_arr, n_mels=self.n_mels)
            elif self.eval_set == "wsj":
                result = subprocess.run(
                    self.audio_files[index],
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                audio_bytes = io.BytesIO(result.stdout)
                audio_arr, _ = torchaudio.load(audio_bytes)
                audio_arr = audio_arr.squeeze(0)
                audio_arr = audio.pad_or_trim(audio_arr)
                audio_arr = audio_arr.float()
                audio_input = audio.log_mel_spectrogram(audio_arr, n_mels=self.n_mels)
                text_y = self.transcript_texts[index]
                audio_fp = ""
            elif self.eval_set == "callhome" or self.eval_set == "switchboard":
                audio_fp, start_time, end_time = self.audio_files[index]
                # num_frames = int(end_time * 8000) - int(start_time * 8000)
                # audio_arr, _ = torchaudio.load(audio_fp, frame_offset=int(start_time * 8000), num_frames=num_frames)
                # # audio_arr = audio_arr.mean(dim=0, keepdim=True)
                # if channel == "A":
                #     audio_arr = audio_arr[0, :]
                # elif channel == "B":
                #     audio_arr = audio_arr[1, :]
                # audio_arr = audio_arr.squeeze(0)
                # audio_arr = audio.pad_or_trim(audio_arr)
                # audio_arr = audio_arr.float()
                # audio_input = audio.log_mel_spectrogram(audio_arr, n_mels=self.n_mels)
                audio_arr, audio_input = self.preprocess_audio(
                    audio_fp, sr=16000, start_time=start_time, end_time=end_time
                )
                text_y = self.transcript_texts[index]
            else:
                audio_fp = self.audio_files[index]
                audio_arr, audio_input = self.preprocess_audio(audio_fp)
                text_y = self.transcript_texts[index]

            return audio_fp, audio_arr, audio_input, text_y
        elif self.task == "long_form_transcribe":
            if self.eval_set == "tedlium":
                waveform = self.dataset[index]["audio"]["array"]
                sampling_rate = self.dataset[index]["audio"]["sampling_rate"]
                text_y = self.dataset[index]["text"]

                if sampling_rate != 16000:
                    waveform = librosa.resample(
                        waveform, orig_sr=sampling_rate, target_sr=16000
                    )

                # audio_arr = audio.pad_or_trim(waveform)
                # audio_arr = audio_arr.astype(np.float32)
                audio_arr = waveform.astype(np.float32)
                audio_input = ""

                # audio_arr, _, text_y, talk_id, speaker_id, identifier = self.dataset[
                #     index
                # ]
                # audio_input = ""
                # audio_fp = "_".join([talk_id, speaker_id, identifier])
            elif self.eval_set == "meanwhile":
                waveform = self.dataset[index]["audio"]["array"]
                sampling_rate = self.dataset[index]["audio"]["sampling_rate"]
                text_y = self.dataset[index]["text"]

                if sampling_rate != 16000:
                    waveform = librosa.resample(
                        waveform, orig_sr=sampling_rate, target_sr=16000
                    )

                audio_arr = waveform.astype(np.float32)
                audio_input = ""
            elif self.eval_set == "rev16":
                waveform = self.dataset[index]["audio"]["array"]
                sampling_rate = self.dataset[index]["audio"]["sampling_rate"]
                text_y = self.dataset[index]["transcription"]

                if sampling_rate != 16000:
                    waveform = librosa.resample(
                        waveform, orig_sr=sampling_rate, target_sr=16000
                    )

                audio_arr = waveform.astype(np.float32)
                audio_input = ""
            elif self.eval_set == "earnings21":
                waveform = self.dataset[index]["audio"]["array"]
                sampling_rate = self.dataset[index]["audio"]["sampling_rate"]
                text_y = self.dataset[index]["transcription"]

                if sampling_rate != 16000:
                    waveform = librosa.resample(
                        waveform, orig_sr=sampling_rate, target_sr=16000
                    )

                audio_arr = waveform.astype(np.float32)
                audio_input = ""
            elif self.eval_set == "earnings22":
                waveform = self.dataset[index]["audio"]["array"]
                sampling_rate = self.dataset[index]["audio"]["sampling_rate"]
                text_y = self.dataset[index]["transcription"]

                if sampling_rate != 16000:
                    waveform = librosa.resample(
                        waveform, orig_sr=sampling_rate, target_sr=16000
                    )

                audio_arr = waveform.astype(np.float32)
                audio_input = ""
            elif self.eval_set == "coraal":
                waveform, sampling_rate = torchaudio.load(self.audio_files[index])
                waveform = waveform.squeeze(0).cpu().numpy()
                text_y = self.transcript_texts[index]

                if sampling_rate != 16000:
                    waveform = librosa.resample(
                        waveform, orig_sr=sampling_rate, target_sr=16000
                    )

                audio_arr = waveform.astype(np.float32)
                audio_input = ""
            elif self.eval_set == "kincaid46":
                waveform, sampling_rate = torchaudio.load(self.audio_files[index])
                waveform = waveform.squeeze(0).cpu().numpy()
                text_y = self.transcript_texts[index]

                if sampling_rate != 16000:
                    waveform = librosa.resample(
                        waveform, orig_sr=sampling_rate, target_sr=16000
                    )

                audio_arr = waveform.astype(np.float32)
                audio_input = ""
            return audio_fp, audio_arr, audio_input, text_y
        elif self.task == "ml_transcribe":
            pass
        elif self.task == "translate":
            pass
        elif self.task == "lang_id":
            if self.eval_set == "fleurs":
                waveform = self.dataset[index]["audio"]["array"]
                sampling_rate = self.dataset[index]["audio"]["sampling_rate"]
                language = self.dataset[index]["language"]
                lang_id = FLEURS_LANG_TO_ID[language]

                if sampling_rate != 16000:
                    waveform = librosa.resample(
                        waveform, orig_sr=sampling_rate, target_sr=16000
                    )

                audio_arr = audio.pad_or_trim(waveform)
                audio_arr = audio_arr.astype(np.float32)
                audio_input = audio.log_mel_spectrogram(audio_arr, n_mels=self.n_mels)

            return audio_fp, audio_arr, audio_input, lang_id

    def preprocess_audio(self, audio_file, sr=16000, start_time=None, end_time=None):
        audio_arr = audio.load_audio(audio_file, sr=sr)
        if start_time is not None and end_time is not None:
            audio_arr = audio_arr[int(start_time * sr) : int(end_time * sr)]
        audio_arr = audio.pad_or_trim(audio_arr)
        mel_spec = audio.log_mel_spectrogram(audio_arr, n_mels=self.n_mels)
        return audio_arr, mel_spec


def short_form_eval(
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
        "coraal",
        "chime6",
        "wsj",
        "callhome",
        "switchboard",
    ],
    log_dir: str,
    n_mels: int = 80,
    current_step: Optional[int] = None,
    train_exp_name: Optional[str] = None,
    train_run_id: Optional[str] = None,
    wandb_log: bool = False,
    wandb_log_dir: Optional[str] = None,
    run_id_dir: Optional[str] = None,
    eval_dir: str = "data/eval",
    hf_token: Optional[str] = None,
    cuda: bool = True,
    bootstrap: bool = False,
):
    if "inf" not in ckpt and ckpt.split("/")[-2] != "whisper_ckpts":
        ckpt = gen_inf_ckpt(ckpt, ckpt.replace(".pt", "_inf.pt"))

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(wandb_log_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device("cuda") if cuda else torch.device("cpu")

    dataset = EvalDataset(
        task="eng_transcribe",
        eval_set=eval_set,
        hf_token=hf_token,
        eval_dir=eval_dir,
        n_mels=n_mels,
    )
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
    if bootstrap:
        per_sample_wer = []

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
        if train_exp_name is not None:
            if not os.path.exists(f"{run_id_dir}/{train_exp_name}_eval.txt"):
                run_id = wandb.util.generate_id()
                with open(f"{run_id_dir}/{train_exp_name}_eval.txt", "w") as f:
                    f.write(run_id)
            else:
                with open(f"{run_id_dir}/{train_exp_name}_eval.txt", "r") as f:
                    run_id = f.read().strip()
        else:
            run_id = wandb.util.generate_id()

        if ckpt.split("/")[-3] == "ow_ckpts":
            ow_or_w = "open-whisper"
        elif ckpt.split("/")[-3] == "yodas":
            ow_or_w = "yodas"
        elif ckpt.split("/")[-3] == "owsm":
            ow_or_w = "owsm"
        else:
            ow_or_w = "whisper"
        exp_name = f"{eval_set}_eval" if ow_or_w == "whisper" else f"ow_{eval_set}_eval"
        model_sizes = ["tiny", "small", "base", "medium", "large"]
        model_size = [model_size for model_size in model_sizes if model_size in ckpt][0]
        config = {
            "ckpt": "/".join(ckpt.split("/")[-2:]),
            "model": ow_or_w,
            "model_size": model_size,
        }
        if train_run_id is not None:
            config["train_run_id"] = train_run_id

        wandb.init(
            id=run_id,
            resume="allow",
            project="open_whisper",
            entity="dogml",
            job_type="evals",
            name=exp_name if train_exp_name is None else train_exp_name,
            dir=wandb_log_dir,
            config=config,
            tags=["eval", eval_set, ow_or_w, model_size],
        )
        eval_table = wandb.Table(columns=wandb_table_cols)

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            audio_fp, audio_arr, audio_input, text_y = batch
            # print(f"{audio_fp=}")
            # print(f"{audio_arr.shape=}")
            # print(f"{text_y=}")
            # print(f"{audio_input.shape=}")

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
            # print(f"{norm_pred_text=}")
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
            # print(f"{norm_tgt_text=}")

            # break

            references.extend(norm_tgt_text)
            hypotheses.extend(norm_pred_text)

            if wandb_log and (batch_idx + 1) // 10 == 1:
                for i, text in enumerate(norm_pred_text):
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
                        eval_set,
                        wandb.Audio(audio_arr[i], sample_rate=16000),
                        norm_pred_text[i],
                        norm_tgt_text[i],
                        subs,
                        dels,
                        ins,
                        wer,
                    )

                if train_run_id is not None:
                    wandb.log({f"eval_table_{current_step}": eval_table})
                else:
                    wandb.log({f"eval_table": eval_table})
            elif bootstrap:
                per_sample_wer.extend(
                    [
                        [
                            jiwer.wer(
                                reference=norm_tgt_text[i], hypothesis=norm_pred_text[i]
                            ),
                            len(norm_tgt_text[i]),
                        ]
                        for i in range(len(norm_pred_text))
                    ]
                )

    avg_wer = jiwer.wer(references, hypotheses) * 100
    avg_measures = jiwer.compute_measures(truth=references, hypothesis=hypotheses)
    avg_subs = avg_measures["substitutions"]
    avg_ins = avg_measures["insertions"]
    avg_dels = avg_measures["deletions"]

    if bootstrap:
        with open(f"{log_dir}/{eval_set}_sample_wer.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["wer", "ref_length"])
            for wer, length in per_sample_wer:
                writer.writerow([wer, length])

    print(
        f"{eval_set} WER: {avg_wer}, Average Subs: {avg_subs}, Average Ins: {avg_ins}, Average Dels: {avg_dels}"
    )

    if wandb_log and train_run_id is not None:
        wandb.log({f"eval/{eval_set}_wer": avg_wer, "global_step": current_step})
        wandb.log({f"eval/{eval_set}_subs": avg_subs, "global_step": current_step})
        wandb.log({f"eval/{eval_set}_ins": avg_ins, "global_step": current_step})
        wandb.log({f"eval/{eval_set}_dels": avg_dels, "global_step": current_step})
    elif not wandb_log and train_run_id is not None:
        with open(f"{log_dir}/{train_exp_name}_{train_run_id}.txt", "a") as f:
            f.write(
                f"Current step {current_step}, {eval_set} WER: {avg_wer}, Subs: {avg_subs}, Ins: {avg_ins}, Dels: {avg_dels}\n"
            )
    elif wandb_log and train_run_id is None:
        wandb.run.summary["avg_wer"] = avg_wer
        wandb.run.summary["avg_subs"] = avg_subs
        wandb.run.summary["avg_ins"] = avg_ins
        wandb.run.summary["avg_dels"] = avg_dels
    elif not wandb_log and train_run_id is None:
        with open(f"{log_dir}/eval_results.txt", "a") as f:
            f.write(
                f"{eval_set} WER: {avg_wer}, Subs: {avg_subs}, Ins: {avg_ins}, Dels: {avg_dels}\n"
            )

    if train_run_id is not None:
        os.remove(ckpt)

    return None


def hf_eval(
    batch_size: int,
    num_workers: int,
    model: str,
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
        "coraal",
        "chime6",
        "wsj",
        "callhome",
        "switchboard",
    ],
    log_dir: str,
    n_mels: int = 80,
    wandb_log: bool = False,
    wandb_log_dir: Optional[str] = None,
    eval_dir: str = "data/eval",
    hf_token: Optional[str] = None,
    cuda: bool = True,
    bootstrap: bool = False,
):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(wandb_log_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device("cuda") if cuda else torch.device("cpu")

    dataset = EvalDataset(
        task="eng_transcribe",
        eval_set=eval_set,
        hf_token=hf_token,
        eval_dir=eval_dir,
        n_mels=n_mels,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    model_name = model.replace("/", "_").replace("-", "_")

    if "nvidia" not in model:
        processor = AutoProcessor.from_pretrained(
            model, trust_remote_code=True, token=hf_token
        )
        
    if "seamless" in model:
        model = AutoModel.from_pretrained(model, trust_remote_code=True)
    elif "phi" in model:
        model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
        generation_config = GenerationConfig.from_pretrained(model, 'generation_config.json')
    elif "Qwen" in model:
        model = AutoModelForSeq2SeqLM.from_pretrained(model, trust_remote_code=True)
    elif "nvidia" in model:
        compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = ASRModel.from_pretrained(model, map_location=device)
        model.to(compute_dtype).eval()

    normalizer = EnglishTextNormalizer()

    hypotheses = []
    references = []
    if bootstrap:
        per_sample_wer = []

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
        run_id = wandb.util.generate_id()

        exp_name = f"hf_{model_name}_{eval_set}_eval"

        wandb.init(
            id=run_id,
            resume="allow",
            project="open_whisper",
            entity="dogml",
            job_type="evals",
            name=exp_name,
            dir=wandb_log_dir,
            tags=["eval", eval_set, "hf"],
        )
        eval_table = wandb.Table(columns=wandb_table_cols)

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            audio_fp, audio_arr, audio_input, text_y = batch
            audio_arr = audio_arr.to(device)
            # print(f"{audio_fp=}")
            # print(f"{audio_arr.shape=}")
            # print(f"{text_y=}")
            # print(f"{audio_input.shape=}")

            norm_tgt_text = [normalizer(text) for text in text_y]
            
            # input_values = processor(
            #         audio_arr, return_tensors="pt", padding="longest"
            #     ).input_values
            #     input_values = input_values.squeeze(0).to(device)
            #     with torch.no_grad():
            #         logits = model(input_values).logits
            #     predicted_ids = torch.argmax(logits, dim=-1)
            #     results = processor.batch_decode(predicted_ids)

            if "seamless" in model_name:
                audio_inputs = processor(audios=audio_arr, return_tensors="pt").to(device)
                output_tokens = model.generate(**audio_inputs, tgt_lang="eng", generate_speech=False)
                results = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
            elif "phi" in model_name:
                user_prompt = '<|user|>'
                assistant_prompt = '<|assistant|>'
                prompt_suffix = '<|end|>'
                speech_prompt = "Based on the attached audio, generate a comprehensive text transcription of the spoken content."
                prompt = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'

                audio_arr = list(torch.unbind(audio_arr, dim=0))
                inputs = processor(text=prompt, audios=audio_arr, return_tensors='pt').to('cuda:0')
                generate_ids = model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    generation_config=generation_config,
                )
                generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
                results = processor.batch_decode(
                    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            elif "Qwen" in model_name:
                audio_arr = list(torch.unbind(audio_arr, dim=0))
                prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Detect the language and recognize the speech: <|en|>"
                audio_inputs = processor(text=prompt, audios=audio_arr, sampling_rate=16000, return_tensors="pt", padding=True)
                
                output_ids = model.generate(**inputs, max_new_tokens=256, min_new_tokens=1, do_sample=False)
                output_ids = output_ids[:, inputs.input_ids.size(1):]
                results = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            elif "nvidia" in model_name:
                audio_arr = list(torch.unbind(audio_arr, dim=0))
                with autocast(
                    enabled=False, dtype=compute_dtype
                ), torch.inference_mode(), torch.no_grad():
                    if "canary" in model_name:
                        results = model.transcribe(
                            audio_arr,
                            batch_size=batch_size,
                            verbose=False,
                            pnc="no",
                            num_workers=num_workers,
                        )
                    else:
                        results = model.transcribe(
                            audio_arr,
                            batch_size=batch_size,
                            verbose=False,
                            num_workers=num_workers,
                        )
                results = [result.text for result in results]

            norm_pred_text = [
                normalizer(results[i])
                for i in range(len(results))
                if norm_tgt_text[i] != ""
                and norm_tgt_text[i] != "ignore time segment in scoring"
            ]
            # print(f"{norm_pred_text=}")
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
            # print(f"{norm_tgt_text=}")

            # break

            references.extend(norm_tgt_text)
            hypotheses.extend(norm_pred_text)

            if wandb_log and (batch_idx + 1) // 10 == 1:
                for i, text in enumerate(norm_pred_text):
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
                        eval_set,
                        wandb.Audio(audio_arr[i], sample_rate=16000),
                        norm_pred_text[i],
                        norm_tgt_text[i],
                        subs,
                        dels,
                        ins,
                        wer,
                    )

                wandb.log({f"eval_table": eval_table})
            elif bootstrap:
                per_sample_wer.extend(
                    [
                        [
                            jiwer.wer(
                                reference=norm_tgt_text[i], hypothesis=norm_pred_text[i]
                            ),
                            len(norm_tgt_text[i]),
                        ]
                        for i in range(len(norm_pred_text))
                    ]
                )

    avg_wer = jiwer.wer(references, hypotheses) * 100
    avg_measures = jiwer.compute_measures(truth=references, hypothesis=hypotheses)
    avg_subs = avg_measures["substitutions"]
    avg_ins = avg_measures["insertions"]
    avg_dels = avg_measures["deletions"]

    if bootstrap:
        with open(f"{log_dir}/{eval_set}_sample_wer.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["wer", "ref_length"])
            for wer, length in per_sample_wer:
                writer.writerow([wer, length])

    print(
        f"{eval_set} WER: {avg_wer}, Average Subs: {avg_subs}, Average Ins: {avg_ins}, Average Dels: {avg_dels}"
    )

    if wandb_log:
        wandb.run.summary["avg_wer"] = avg_wer
        wandb.run.summary["avg_subs"] = avg_subs
        wandb.run.summary["avg_ins"] = avg_ins
        wandb.run.summary["avg_dels"] = avg_dels
    else:
        with open(f"{log_dir}/eval_results.txt", "a") as f:
            f.write(
                f"{eval_set} WER: {avg_wer}, Subs: {avg_subs}, Ins: {avg_ins}, Dels: {avg_dels}\n"
            )

    return None


def long_form_eval(
    batch_size: int,
    num_workers: int,
    ckpt: str,
    eval_set: Literal[
        "tedlium",
        "meanwhile",
        "rev16",
        "earnings21",
        "earnings22",
        "coraal",
        "kincaid46",
    ],
    log_dir: str,
    n_mels: int = 80,
    bootstrap: bool = False,
    exp_name: Optional[str] = None,
    wandb_log: bool = False,
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

    dataset = EvalDataset(
        task="long_form_transcribe",
        eval_set=eval_set,
        hf_token=hf_token,
        eval_dir=eval_dir,
        n_mels=n_mels,
    )
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
    if bootstrap:
        per_sample_wer = []

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

        run_id = wandb.util.generate_id()
        if ckpt.split("/")[-3] == "ow_ckpts":
            ow_or_w = "open-whisper"
        elif ckpt.split("/")[-3] == "yodas":
            ow_or_w = "yodas"
        elif ckpt.split("/")[-3] == "owsm":
            ow_or_w = "owsm"
        else:
            ow_or_w = "whisper"
        exp_name = f"{eval_set}_eval" if ow_or_w == "whisper" else f"ow_{eval_set}_eval"
        model_sizes = ["tiny", "small", "base", "medium", "large"]
        model_size = [model_size for model_size in model_sizes if model_size in ckpt][0]
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
            _, audio_arr, _, text_y = batch

            norm_tgt_text = [normalizer(text) for text in text_y]
            audio_arr = audio_arr.to(device)
            print(f"{audio_arr.shape=}")

            options = dict(
                task="transcribe",
                language="en",
                without_timestamps=False,
                beam_size=5,
                best_of=5,
            )
            results = model.transcribe(audio_arr[0], verbose=False, **options)

            norm_pred_text = [
                normalizer(results["text"])
                for i in range(len(norm_tgt_text))
                if norm_tgt_text[i] != ""
            ]
            if wandb_log:
                audio_arr = [
                    audio_arr.cpu().numpy()[i]
                    for i in range(len(norm_tgt_text))
                    if norm_tgt_text[i] != ""
                ]
            norm_tgt_text = [
                norm_tgt_text[i]
                for i in range(len(norm_tgt_text))
                if norm_tgt_text[i] != ""
            ]

            print(f"{norm_pred_text=}")
            print(f"{norm_tgt_text=}")

            references.extend(norm_tgt_text)
            hypotheses.extend(norm_pred_text)

            if wandb_log and not bootstrap:
                for i, text in enumerate(norm_pred_text):
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
                        eval_set,
                        wandb.Audio(audio_arr[i], sample_rate=16000),
                        norm_pred_text[i],
                        norm_tgt_text[i],
                        subs,
                        dels,
                        ins,
                        wer,
                    )
            elif bootstrap:
                per_sample_wer.extend(
                    [
                        [
                            jiwer.wer(
                                reference=norm_tgt_text[i], hypothesis=norm_pred_text[i]
                            ),
                            len(norm_tgt_text[i]),
                        ]
                        for i in range(len(norm_pred_text))
                    ]
                )
            else:
                wer = (
                    np.round(
                        jiwer.wer(
                            reference=norm_tgt_text,
                            hypothesis=norm_pred_text,
                        ),
                        2,
                    )
                    * 100
                )
                print(f"{wer=}")
                # with open(f"{eval_set}_eval_results.txt", "a") as f:
                #     f.write(norm_pred_text[0] + "\n")
                #     f.write(norm_tgt_text[0] + "\n")
                #     f.write("\n")

                # break

        if wandb_log:
            wandb.log({f"eval_table": eval_table})

    avg_wer = jiwer.wer(references, hypotheses) * 100
    avg_measures = jiwer.compute_measures(truth=references, hypothesis=hypotheses)
    avg_subs = avg_measures["substitutions"]
    avg_ins = avg_measures["insertions"]
    avg_dels = avg_measures["deletions"]

    if bootstrap:
        with open(f"{log_dir}/{eval_set}_sample_wer.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["wer", "ref_length"])
            for wer, length in per_sample_wer:
                writer.writerow([wer, length])

    if wandb_log:
        wandb.run.summary["avg_wer"] = avg_wer
        wandb.run.summary["avg_subs"] = avg_subs
        wandb.run.summary["avg_ins"] = avg_ins
        wandb.run.summary["avg_dels"] = avg_dels

    print(
        f"Average WER: {avg_wer}, Average Subs: {avg_subs}, Average Ins: {avg_ins}, Average Dels: {avg_dels}"
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
            task="ml_transcribe",
            eval_set=eval_set,
            lang=lang,
            hf_token=hf_token,
            eval_dir=eval_dir,
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


def translate_eval():
    pass


def lang_id_eval(
    batch_size: int,
    num_workers: int,
    ckpt: str,
    eval_set: Literal["fleurs",],
    log_dir: str,
    lang: Optional[str] = None,
    wandb_log: bool = False,
    wandb_log_dir: str = "wandb",
    eval_dir: str = "data/eval",
    cuda: bool = True,
):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(wandb_log_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device("cuda") if cuda else torch.device("cpu")
    tokenizer = get_tokenizer(multilingual=True)

    if ckpt == "voxlingua107":
        if cuda:
            model = EncoderClassifier.from_hparams(
                source="speechbrain/lang-id-voxlingua107-ecapa",
                savedir="tmp",
                run_opts={"device": "cuda"},
            )
        else:
            model = EncoderClassifier.from_hparams(
                source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp"
            )
    else:
        if "inf" not in ckpt and ckpt.split("/")[-2] != "whisper_ckpts":
            ckpt = gen_inf_ckpt(ckpt, ckpt.replace(".pt", "_inf.pt"))

        model = load_model(name=ckpt, device=device, inference=True, in_memory=True)
        model.eval()

    dataset = EvalDataset(
        task="lang_id", eval_set=eval_set, lang=lang, eval_dir=eval_dir
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    if wandb_log:
        wandb.init(
            id=wandb.util.generate_id(),
            resume="allow",
            project="open_whisper",
            entity="dogml",
            save_code=True,
            settings=wandb.Settings(init_timeout=300, _service_wait=300),
        )

    correct_count = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            _, audio_arr, audio_input, lang_ids = batch

            audio_input = audio_input.to(device)

            if ckpt == "voxlingua107":
                results = model.classify_batch(audio_arr)
                pred_lang_ids = [res.split(": ")[0] for res in results[3]]
                pred_lang_ids = [
                    VOX_TO_FLEURS[lang_id] if lang_id in VOX_TO_FLEURS else lang_id
                    for lang_id in pred_lang_ids
                ]
                print(f"{pred_lang_ids=}")
            else:
                results = detect_language(model, audio_input)

                pred_lang_ids_str = tokenizer.decode(results[0])
                pred_lang_ids = re.findall(r"<\|(.*?)\|>", pred_lang_ids_str)
                pred_lang_ids = [
                    OW_TO_FLEURS[lang_id] if lang_id in OW_TO_FLEURS else lang_id
                    for lang_id in pred_lang_ids
                ]
                print(f"{pred_lang_ids=}")
            print(f"{lang_ids=}")

            for i in range(len(lang_ids)):
                if lang_ids[i] == "en":
                    if lang_ids[i] == pred_lang_ids[i]:
                        true_positive += 1
                        correct_count += 1
                    else:
                        false_negative += 1
                else:
                    if pred_lang_ids[i] != "en":
                        true_negative += 1
                        correct_count += 1
                    else:
                        false_positive += 1
            total += len(lang_ids)
            print(f"{correct_count=}, {total=}")
            print(
                f"{true_positive=}, {true_negative=}, {false_positive=}, {false_negative=}"
            )
            print(f"Accuracy: {100 * (correct_count / total):.2f}")
            if true_positive + false_positive != 0:
                print(
                    f"Precision: {100 * (true_positive / (true_positive + false_positive)):.2f}"
                )
            if true_positive + false_negative != 0:
                print(
                    f"Recall: {100 * (true_positive / (true_positive + false_negative)):.2f}"
                )

    accuracy = 100 * (correct_count / total)
    precision = 100 * (true_positive / (true_positive + false_positive))
    recall = 100 * (true_positive / (true_positive + false_negative))
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    with open(f"{log_dir}/{eval_set}_lang_id_accuracy.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}")
        f.write(f"Precision: {precision}")
        f.write(f"Recall: {recall}")


if __name__ == "__main__":
    Fire(
        {
            "short_form_eval": short_form_eval,
            "ml_eval": ml_eval,
            "long_form_eval": long_form_eval,
            "hf_eval": hf_eval,
            # "lang_id_eval": lang_id_eval,
        }
    )

    # long_form_eval(batch_size=1, num_workers=12, ckpt="/weka/huongn/ow_ckpts/filtered/tagged_data/text_heurs_seg_edit_dist_0.7_edit_dist_0.5_long_tiny_15e4_440K_bs64_ebs512_16workers_5pass_TimestampOn_evalbs8_042525_9zd7k10y/latesttrain_00524288_tiny_ddp-train_grad-acc_fp16_non_ddp_inf.pt", eval_set="tedlium", log_dir="/stage", wandb_log=False, wandb_log_dir="/stage", eval_dir="/weka/huongn/ow_eval", hf_token="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")
    # long_form_eval(batch_size=1, num_workers=1, ckpt="/weka/huongn/ow_ckpts/filtered/tagged_data/text_heurs_seg_edit_dist_0.7_edit_dist_0.5_long_tiny_15e4_440K_bs64_ebs512_16workers_5pass_TimestampOn_evalbs8_042525_9zd7k10y/latesttrain_00524288_tiny_ddp-train_grad-acc_fp16_non_ddp_inf.pt", eval_set="coraal", log_dir="/stage", wandb_log=False, wandb_log_dir="/stage", eval_dir="/weka/huongn/ow_eval", hf_token="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")
    # short_form_eval(
    #     batch_size=96,
    #     num_workers=12,
    #     # ckpt="/weka/huongn/whisper_ckpts/tiny.en.pt",
    #     ckpt="/weka/huongn/ow_ckpts/filtered/tagged_data/text_heurs_seg_edit_dist_0.7_edit_dist_0.5_long_tiny_15e4_440K_bs64_ebs512_16workers_5pass_TimestampOn_evalbs8_042525_9zd7k10y/latesttrain_00524288_tiny_ddp-train_grad-acc_fp16_non_ddp_inf.pt",
    #     eval_set="wsj",
    #     log_dir="/stage",
    #     wandb_log=False,
    #     wandb_log_dir="/stage",
    #     eval_dir="/weka/huongn/ow_eval",
    #     hf_token="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",
    # )
