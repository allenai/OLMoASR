import subprocess
import os
from typing import Literal, Optional
from datasets import load_dataset
from fire import Fire
import shutil

AMI_IDS = [
    "ES2002a",
    "ES2002b",
    "ES2002c",
    "ES2002d",
    "ES2003a",
    "ES2003b",
    "ES2003c",
    "ES2003d",
    "ES2005a",
    "ES2005b",
    "ES2005c",
    "ES2005d",
    "ES2006a",
    "ES2006b",
    "ES2006c",
    "ES2006d",
    "ES2007a",
    "ES2007b",
    "ES2007c",
    "ES2007d",
    "ES2008a",
    "ES2008b",
    "ES2008c",
    "ES2008d",
    "ES2009a",
    "ES2009b",
    "ES2009c",
    "ES2009d",
    "ES2010a",
    "ES2010b",
    "ES2010c",
    "ES2010d",
    "ES2012a",
    "ES2012b",
    "ES2012c",
    "ES2012d",
    "ES2013a",
    "ES2013b",
    "ES2013c",
    "ES2013d",
    "ES2014a",
    "ES2014b",
    "ES2014c",
    "ES2014d",
    "ES2015a",
    "ES2015b",
    "ES2015c",
    "ES2015d",
    "ES2016a",
    "IS1000a",
    "ES2016b",
    "IS1000b",
    "ES2016c",
    "IS1000c",
    "ES2016d",
    "IS1000d",
    "IS1001a",
    "IS1001b",
    "IS1001c",
    "IS1001d",
    "IS1002b",
    "IS1002c",
    "IS1002d",
    "IS1003a",
    "IS1003b",
    "IS1003c",
    "IS1003d",
    "IS1004a",
    "IS1004b",
    "IS1004c",
    "IS1004d",
    "IS1005a",
    "IS1005b",
    "IS1005c",
    "IS1006a",
    "IS1006b",
    "IS1006c",
    "IS1006d",
    "IS1007a",
    "TS3005a",
    "IS1007b",
    "TS3005b",
    "IS1007c",
    "TS3005c",
    "IS1007d",
    "TS3005d",
    "TS3006a",
    "TS3006b",
    "TS3006c",
    "TS3006d",
    "TS3007a",
    "TS3007b",
    "TS3007c",
    "TS3007d",
    "TS3008a",
    "TS3008b",
    "TS3008c",
    "TS3008d",
    "TS3009a",
    "TS3009b",
    "TS3009c",
    "TS3009d",
    "TS3010a",
    "TS3010b",
    "TS3010c",
    "TS3010d",
    "TS3011a",
    "TS3011b",
    "TS3011c",
    "TS3011d",
    "TS3012a",
    "TS3012b",
    "TS3012c",
    "TS3012d",
    "EN2001a",
    "EN2001b",
    "EN2001e",
    "EN2001d",
    "EN2003a",
    "EN2004a",
    "EN2005a",
    "EN2006a",
    "EN2006b",
    "EN2009b",
    "EN2009c",
    "EN2009d",
    "IN1001",
    "IN1002",
    "IN1005",
    "IN1007",
    "IN1008",
    "IN1009",
    "IN1012",
    "IN1013",
    "IN1014",
    "IN1016",
]


def get_eval_train(
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
    eval_dir: str = "data/eval",
    hf_token: Optional[str] = None,
) -> Optional[str]:
    """Download evaluation set

    Downloads specified evaluation set and extracts it to the specified directory. Default directory is `data/eval`.\n
    For CommonVoice specifically, you'll have to download the entire dataset which is around 54GB.
    You'll need to provide your HuggingFace user authentication token to download the dataset.
    To get the token, follow the instructions at `https://huggingface.co/docs/hub/en/security-tokens`.

    Args:
        eval_set: Evaluation set to download
        eval_dir: Directory to download the evaluation set
        hf_token: HuggingFace user authentication token

    """
    os.makedirs(eval_dir, exist_ok=True)
    if eval_set == "librispeech_clean" and not os.path.exists(
        f"{eval_dir}/librispeech_train_clean"
    ):
        # downloading the file
        command = [
            "wget",
            "-P",
            eval_dir,
            "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        ]
        subprocess.run(command)
        # extracting the file
        command = ["tar", "-xvf", f"{eval_dir}/train-clean-100.tar.gz", "-C", eval_dir]
        subprocess.run(command)
        # removing the tar file
        os.remove(f"{eval_dir}/train-clean-100.tar.gz")
        # making test-clean main data folder
        os.rename(
            f"{eval_dir}/LibriSpeech/train-clean-100", f"{eval_dir}/librispeech_train_clean"
        )
        shutil.rmtree(f"{eval_dir}/LibriSpeech")
    elif eval_set == "librispeech_other":
        # downloading the file
        command = [
            "wget",
            "-P",
            eval_dir,
            "https://www.openslr.org/resources/12/train-other-500.tar.gz",
        ]
        subprocess.run(command)
        # extracting the file
        command = ["tar", "-xvf", f"{eval_dir}/train-other-500.tar.gz", "-C", eval_dir]
        subprocess.run(command)
        # removing the tar file
        os.remove(f"{eval_dir}/train-other-500.tar.gz")
        # making test-other main data folder
        os.rename(
            f"{eval_dir}/LibriSpeech/train-other-500", f"{eval_dir}/librispeech_train_other"
        )
        shutil.rmtree(f"{eval_dir}/LibriSpeech")
    elif eval_set == "multilingual_librispeech":
        eval_dir = f"{eval_dir}/mls"
        os.makedirs(eval_dir, exist_ok=True)
        command = [
            "wget",
            "-P",
            eval_dir,
            f"https://dl.fbaipublicfiles.com/mls/mls_{lang}_opus.tar.gz",
        ]
        subprocess.run(command)
        # extracting the file
        command = [
            "tar",
            "-xzvf",
            f"{eval_dir}/mls_{lang}_opus.tar.gz",
            "-C",
            eval_dir,
            f"mls_{lang}_opus/test",
        ]
        subprocess.run(command)
        # removing the tar file
        os.remove(f"{eval_dir}/mls_{lang}_opus.tar.gz")
    elif eval_set == "artie_bias_corpus":
        # downloading the file
        command = [
            "wget",
            "-P",
            eval_dir,
            "http://ml-corpora.artie.com/artie-bias-corpus.tar.gz",
        ]
        subprocess.run(command)
        # extracting the file
        command = [
            "tar",
            "-xvf",
            f"{eval_dir}/artie-bias-corpus.tar.gz",
            "-C",
            eval_dir,
        ]
        subprocess.run(command)
        # removing the tar file
        os.remove(f"{eval_dir}/artie-bias-corpus.tar.gz")
    elif eval_set == "fleurs":
        dataset = load_dataset(
            path="google/fleurs",
            name="en_us",
            split="test",
            cache_dir=eval_dir,
            trust_remote_code=True,
            num_proc=15,
            save_infos=True,
        )
    elif eval_set == "tedlium":
        # downloading the files
        command = [
            "wget",
            "-P",
            eval_dir,
            "https://huggingface.co/datasets/LIUM/tedlium/resolve/main/TEDLIUM_release3/legacy/train_1.tar.gz",
        ]
        subprocess.run(command)
        # extracting the file
        command = ["tar", "-xvf", f"{eval_dir}/train_1.tar.gz", "-C", eval_dir]
        subprocess.run(command)
        # removing the tar file
        os.remove(f"{eval_dir}/train_1.tar.gz")
        # renaming the folder
        os.makedirs(f"{eval_dir}/TEDLIUM_release-3/data", exist_ok=True)
        os.rename(f"{eval_dir}/train", f"{eval_dir}/TEDLIUM_release-3/data")
        os.makedirs(f"{eval_dir}/TEDLIUM_release-3/data/sph", exist_ok=True)
        os.makedirs(f"{eval_dir}/TEDLIUM_release-3/data/stm", exist_ok=True)
        for f in os.listdir(f"{eval_dir}/TEDLIUM_release-3/data"):
            if f.endswith(".stm"):
                os.rename(
                    f"{eval_dir}/TEDLIUM_release-3/data/{f}",
                    f"{eval_dir}/TEDLIUM_release-3/data/stm/{f}",
                )
            elif f.endswith(".sph"):
                os.rename(
                    f"{eval_dir}/TEDLIUM_release-3/data/{f}",
                    f"{eval_dir}/TEDLIUM_release-3/data/sph/{f}",
                )
    elif eval_set == "voxpopuli":
        dataset = load_dataset(
            path="facebook/voxpopuli",
            name="en",
            split="test",
            cache_dir=eval_dir,
            trust_remote_code=True,
            num_proc=15,
            save_infos=True,
        )
    elif eval_set == "common_voice":
        dataset = load_dataset(
            path="mozilla-foundation/common_voice_5_1",
            name="en",
            split="train",
            token=hf_token,
            cache_dir=eval_dir,
            trust_remote_code=True,
            num_proc=15,
            save_infos=True,
        )
    elif eval_set.startswith("ami"):
        ami_dir = f"{eval_dir}/ami"
        os.makedirs(ami_dir, exist_ok=True)
        if eval_set == "ami_ihm":
            ami_ihm_dir = f"{ami_dir}/ihm"
            os.makedirs(ami_ihm_dir, exist_ok=True)
            command = [
                "wget",
                "-P",
                ami_ihm_dir,
                "https://huggingface.co/datasets/edinburghcstr/ami/resolve/main/annotations/train/text",
            ]
            subprocess.run(command)
            for _id in AMI_IDS:
                command = [
                    "wget",
                    "-P",
                    ami_ihm_dir,
                    f"https://huggingface.co/datasets/edinburghcstr/ami/resolve/main/audio/ihm/train/{_id}.tar.gz",
                ]
                subprocess.run(command)
                command = [
                    "tar",
                    "-xvf",
                    f"{ami_ihm_dir}/{_id}.tar.gz",
                    "-C",
                    ami_ihm_dir,
                ]
                subprocess.run(command)
                os.remove(f"{ami_ihm_dir}/{_id}.tar.gz")
        elif eval_set == "ami_sdm":
            ami_sdm_dir = f"{ami_dir}/sdm"
            os.makedirs(ami_sdm_dir, exist_ok=True)
            command = [
                "wget",
                "-P",
                ami_sdm_dir,
                "https://huggingface.co/datasets/edinburghcstr/ami/resolve/main/annotations/train/text",
            ]
            subprocess.run(command)
            for _id in AMI_IDS:
                command = [
                    "wget",
                    "-P",
                    ami_sdm_dir,
                    f"https://huggingface.co/datasets/edinburghcstr/ami/resolve/main/audio/sdm/train/{_id}.tar.gz",
                ]
                subprocess.run(command)
                command = [
                    "tar",
                    "-xvf",
                    f"{ami_sdm_dir}/{_id}.tar.gz",
                    "-C",
                    ami_sdm_dir,
                ]
                subprocess.run(command)
                os.remove(f"{ami_sdm_dir}/{_id}.tar.gz")

            for root, dirs, files in os.walk(ami_sdm_dir):
                for f in files:
                    if "sdm" in f:
                        new_name = f.replace("sdm", "h00")
                        os.rename(f"{root}/{f}", f"{root}/{new_name}")


if __name__ == "__main__":
    Fire(get_eval_train)
