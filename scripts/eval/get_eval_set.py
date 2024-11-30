import subprocess
import os
from typing import Literal, Optional
from datasets import load_dataset
from fire import Fire
import shutil

AMI_IDS = [
    "EN2002a",
    "EN2002b",
    "EN2002c",
    "EN2002d",
    "ES2004a",
    "ES2004b",
    "ES2004c",
    "ES2004d",
    "IS1009a",
    "IS1009b",
    "IS1009c",
    "IS1009d",
    "TS3003a",
    "TS3003b",
    "TS3003c",
    "TS3003d",
]


def get_eval_set(
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
        f"{eval_dir}/librispeech_test_clean"
    ):
        # downloading the file
        command = [
            "wget",
            "-P",
            eval_dir,
            "https://www.openslr.org/resources/12/test-clean.tar.gz",
        ]
        subprocess.run(command)
        # extracting the file
        command = ["tar", "-xvf", f"{eval_dir}/test-clean.tar.gz", "-C", eval_dir]
        subprocess.run(command)
        # removing the tar file
        os.remove(f"{eval_dir}/test-clean.tar.gz")
        # making test-clean main data folder
        os.rename(
            f"{eval_dir}/LibriSpeech/test-clean", f"{eval_dir}/librispeech_test_clean"
        )
        shutil.rmtree(f"{eval_dir}/LibriSpeech")
    elif eval_set == "librispeech_other":
        # downloading the file
        command = [
            "wget",
            "-P",
            eval_dir,
            "https://www.openslr.org/resources/12/test-other.tar.gz",
        ]
        subprocess.run(command)
        # extracting the file
        command = ["tar", "-xvf", f"{eval_dir}/test-other.tar.gz", "-C", eval_dir]
        subprocess.run(command)
        # removing the tar file
        os.remove(f"{eval_dir}/test-other.tar.gz")
        # making test-other main data folder
        os.rename(
            f"{eval_dir}/LibriSpeech/test-other", f"{eval_dir}/librispeech_test_other"
        )
        shutil.rmtree(f"{eval_dir}/LibriSpeech")
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
        # downloading the files
        command = [
            "wget",
            "-P",
            eval_dir,
            "https://huggingface.co/datasets/google/fleurs/resolve/main/data/en_us/test.tsv",
        ]
        subprocess.run(command)
        command = [
            "wget",
            "-P",
            eval_dir,
            "https://huggingface.co/datasets/google/fleurs/resolve/main/data/en_us/audio/test.tar.gz",
        ]
        subprocess.run(command)
        # extracting the file
        command = ["tar", "-xvf", f"{eval_dir}/test.tar.gz", "-C", eval_dir]
        subprocess.run(command)
        # removing the tar file
        os.remove(f"{eval_dir}/test.tar.gz")
        # making directory
        os.makedirs(f"{eval_dir}/fleurs", exist_ok=True)
        os.rename(f"{eval_dir}/test.tsv", f"{eval_dir}/fleurs/test.tsv")
        os.rename(f"{eval_dir}/test", f"{eval_dir}/fleurs/test")
    elif eval_set == "tedlium":
        # downloading the files
        command = [
            "wget",
            "-P",
            eval_dir,
            "https://huggingface.co/datasets/LIUM/tedlium/resolve/main/TEDLIUM_release3/legacy/test.tar.gz",
        ]
        subprocess.run(command)
        # extracting the file
        command = ["tar", "-xvf", f"{eval_dir}/test.tar.gz", "-C", eval_dir]
        subprocess.run(command)
        # removing the tar file
        os.remove(f"{eval_dir}/test.tar.gz")
        # renaming the folder
        os.makedirs(f"{eval_dir}/TEDLIUM_release-3/legacy", exist_ok=True)
        os.rename(f"{eval_dir}/test", f"{eval_dir}/TEDLIUM_release-3/legacy/test")
        os.makedirs(f"{eval_dir}/TEDLIUM_release-3/legacy/test/sph", exist_ok=True)
        os.makedirs(f"{eval_dir}/TEDLIUM_release-3/legacy/test/stm", exist_ok=True)
        for f in os.listdir(f"{eval_dir}/TEDLIUM_release-3/legacy/test"):
            if f.endswith(".stm"):
                os.rename(f"{eval_dir}/TEDLIUM_release-3/legacy/test/{f}", f"{eval_dir}/TEDLIUM_release-3/legacy/test/stm/{f}")
            elif f.endswith(".sph"):
                os.rename(f"{eval_dir}/TEDLIUM_release-3/legacy/test/{f}", f"{eval_dir}/TEDLIUM_release-3/legacy/test/sph/{f}")
    elif eval_set == "voxpopuli":
        command = [
            "wget",
            "-P",
            eval_dir,
            "https://huggingface.co/datasets/facebook/voxpopuli/resolve/main/data/en/test/test_part_0.tar.gz",
        ]
        subprocess.run(command)
        command = [
            "wget",
            "-P",
            eval_dir,
            "https://huggingface.co/datasets/facebook/voxpopuli/resolve/main/data/en/asr_test.tsv",
        ]
        subprocess.run(command)
        # extracting the file
        command = ["tar", "-xvf", f"{eval_dir}/test_part_0.tar.gz", "-C", eval_dir]
        subprocess.run(command)
        # removing the tar file
        os.remove(f"{eval_dir}/test_part_0.tar.gz")
        # making directory
        os.makedirs(f"{eval_dir}/voxpopuli", exist_ok=True)
        os.rename(f"{eval_dir}/asr_test.tsv", f"{eval_dir}/voxpopuli/asr_test.tsv")
        os.rename(f"{eval_dir}/test_part_0", f"{eval_dir}/voxpopuli/test")
    elif eval_set == "common_voice":
        dataset = load_dataset(
            path="mozilla-foundation/common_voice_5_1",
            name="en",
            token=hf_token,
            split="test",
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
                "https://huggingface.co/datasets/edinburghcstr/ami/resolve/main/annotations/eval/text",
            ]
            subprocess.run(command)
            for _id in AMI_IDS:
                command = [
                    "wget",
                    "-P",
                    ami_ihm_dir,
                    f"https://huggingface.co/datasets/edinburghcstr/ami/resolve/main/audio/ihm/eval/{_id}.tar.gz",
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
                "https://huggingface.co/datasets/edinburghcstr/ami/resolve/main/annotations/eval/text",
            ]
            subprocess.run(command)
            for _id in AMI_IDS:
                command = [
                    "wget",
                    "-P",
                    ami_sdm_dir,
                    f"https://huggingface.co/datasets/edinburghcstr/ami/resolve/main/audio/sdm/eval/{_id}.tar.gz",
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
    Fire(get_eval_set)
