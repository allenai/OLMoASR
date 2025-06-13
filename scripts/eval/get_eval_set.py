import subprocess
import os
from typing import Literal, Optional
from datasets import load_dataset
from fire import Fire
import shutil
import glob
import json
import multiprocessing
from tqdm import tqdm
from itertools import repeat
from pydub import AudioSegment
import tarfile
import pandas as pd
import numpy as np

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
        "librispeech_clean",  # short-form eval sets onwards
        "librispeech_other",
        "artie_bias_corpus",
        "tedlium",
        "common_voice",
        "ami_ihm",
        "ami_sdm",
        "chime6",
        "coraal",
        "callhome",
        "switchboard",
        "wsj",
        "meanwhile",  # long-form eval sets onwards
        "rev16",
        "kincaid46",
        "earnings-21",
        "multilingual_librispeech",  # multilingual eval sets onwards
        "fleurs",
        "voxpopuli",
        "common_voice9",
        "covost2",
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
        if lang is None:
            dataset = load_dataset(
                path="google/fleurs",
                name="en_us",
                split="test",
                cache_dir=eval_dir,
                trust_remote_code=True,
                num_proc=15,
                save_infos=True,
            )
        elif lang == "all":
            dataset = load_dataset(
                path="google/fleurs",
                name="all",
                split="test",
                cache_dir=eval_dir,
                trust_remote_code=True,
                num_proc=15,
                save_infos=True,
            )
        else:
            dataset = load_dataset(
                path="google/fleurs",
                name=lang,
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
                os.rename(
                    f"{eval_dir}/TEDLIUM_release-3/legacy/test/{f}",
                    f"{eval_dir}/TEDLIUM_release-3/legacy/test/stm/{f}",
                )
            elif f.endswith(".sph"):
                os.rename(
                    f"{eval_dir}/TEDLIUM_release-3/legacy/test/{f}",
                    f"{eval_dir}/TEDLIUM_release-3/legacy/test/sph/{f}",
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
            split="test",
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
    elif eval_set == "chime6":
        eval_dir = os.path.join(eval_dir, "chime6")
        os.makedirs(eval_dir, exist_ok=True)
        command = [
            "wget",
            "-P",
            eval_dir,
            "https://www.openslr.org/resources/150/CHiME6_eval.tar.gz",
            "https://www.openslr.org/resources/150/CHiME6_transcriptions.tar.gz",
        ]
        subprocess.run(command)
        command = ["tar", "-xvf", f"{eval_dir}/CHiME6_eval.tar.gz", "-C", eval_dir]
        subprocess.run(command)
        command = [
            "tar",
            "-xvf",
            f"{eval_dir}/CHiME6_transcriptions.tar.gz",
            "-C",
            eval_dir,
        ]
        os.remove(f"{eval_dir}/CHiME6_eval.tar.gz")
        os.remove(f"{eval_dir}/CHiME6_transcriptions.tar.gz")

        os.rename(f"{eval_dir}/CHiME6_eval/CHiME6/audio/eval", f"{eval_dir}/audio")
        shutil.rmtree(f"{eval_dir}/CHiME6_eval")
        for p in glob.glob(f"{eval_dir}/audio/*_U*.wav"):
            os.remove(p)
        shutil.rmtree(f"{eval_dir}/transcriptions/transcriptions/dev")
        shutil.rmtree(f"{eval_dir}/transcriptions/transcriptions/train")
        os.rename(
            f"{eval_dir}/transcriptions/transcriptions/eval", f"{eval_dir}/transcripts"
        )
        shutil.rmtree(f"{eval_dir}/transcriptions")

        def timestamp_to_ms(timestamp):
            h, m, s = map(float, timestamp.split(":"))
            return int((h * 3600 + m * 60 + s) * 1000)

        def create_segment(src_dir, dst_dir, seg_dict):
            audio_file = os.path.join(src_dir, seg_dict["audio_file"])
            segment_file = os.path.join(dst_dir, seg_dict["audio_seg_file"])

            os.makedirs(dst_dir, exist_ok=True)
            audio = AudioSegment.from_wav(audio_file)
            start_time = timestamp_to_ms(seg_dict["start_time"])
            end_time = timestamp_to_ms(seg_dict["end_time"])
            clip = audio[start_time:end_time]
            clip.export(segment_file, format="wav")
            return segment_file

        def parallel_create_segment(args):
            return create_segment(*args)

        for p in glob.glob(f"{eval_dir}/transcripts/*.json"):
            with open(p, "r") as f:
                data = json.load(f)

            for d in data:
                start = timestamp_to_ms(d["start_time"])
                end = timestamp_to_ms(d["end_time"])
                d["audio_file"] = f"{d['session_id']}_{d['speaker']}.wav"
                d["audio_seg_file"] = (
                    f"{d['session_id']}_{d['speaker']}_{start:07}_{end:07}.wav"
                )

            with open(p, "w") as f:
                json.dump(data, f)

            with multiprocessing.Pool() as pool:
                res = list(
                    tqdm(
                        pool.imap_unordered(
                            parallel_create_segment,
                            zip(
                                repeat(f"{eval_dir}/audio"),
                                repeat(f"{eval_dir}/segments"),
                                data,
                            ),
                        ),
                        total=len(data),
                    )
                )
    elif eval_set == "coraal":
        eval_dir = os.path.join(eval_dir, "coraal")
        os.makedirs(eval_dir, exist_ok=True)
        command = ["wget", "-P", eval_dir, "-i", "https://tinyurl.com/coraalfiles"]
        subprocess.run(command)
        
        with open(f"{eval_dir}/coraalfiles", "r") as f:
            lines = f.readlines()
            coraal_subs = list({line.strip().split("/")[4] for line in lines[1:]})
            
        for coraal_sub in coraal_subs:
            os.makedirs(os.path.join(eval_dir, coraal_sub), exist_ok=True)
            
        for p in glob.glob(eval_dir + "/*.gz"):
            dest = os.path.join(
                eval_dir,
                str.lower(os.path.basename(p).split("_")[0]),
                os.path.basename(p),
            )
            print(f"{p} -> {dest}")
            os.rename(p, dest)
            
        for p in glob.glob(eval_dir + "/*"):
            if os.path.isdir(p):
                os.makedirs(os.path.join(p, "transcripts"), exist_ok=True)

        def extract_tar_gz(archive_path, output_dir):
            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Open the tar.gz file
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=output_dir)
                print(f"Extracted '{archive_path}' to '{output_dir}'")

        for p in glob.glob(eval_dir + "/*/*.gz"):
            if "textfiles" in p:
                output_dir = os.path.join(os.path.dirname(p), "transcripts")
                extract_tar_gz(p, output_dir)

        def load_coraal_text(project_path):
            file_pattern = os.path.join(project_path, "*metadata*.txt")
            filenames = glob.glob(file_pattern, recursive=True)
            print(filenames)
            metadata = pd.concat(
                [pd.read_csv(filename, sep="\t") for filename in filenames], sort=False
            )

            rows = []

            for sub, file in (
                metadata[["CORAAL.Sub", "CORAAL.File"]].drop_duplicates().values
            ):
                subpath = os.path.join(project_path, sub.lower())
                text_filename = os.path.join(subpath, "transcripts", file + ".txt")

                text = pd.read_csv(text_filename, sep="\t")
                text["pause"] = text.Content.str.contains("(pause \d+(\.\d{1,2})?)")
                text = text[~text.pause]

                for spkr, sttime, content, entime in text[
                    ["Spkr", "StTime", "Content", "EnTime"]
                ].values:
                    row = {
                        "name": spkr,
                        "speaker": spkr,
                        "start_time": sttime,
                        "end_time": entime,
                        "content": content,
                        "filename": text_filename,
                        "source": "coraal",
                        "location": sub,
                        "basefile": file,
                        "interviewee": spkr in file,
                    }
                    rows.append(row)

            df = pd.DataFrame(rows)
            df = df.sort_values(by=["basefile", "start_time"])
            df["line"] = np.arange(len(df))
            df = df[
                [
                    "basefile",
                    "line",
                    "start_time",
                    "end_time",
                    "speaker",
                    "content",
                    "interviewee",
                    "source",
                    "location",
                    "name",
                    "filename",
                ]
            ]
            print("CORAAL full df len ", len(df))
            df.drop_duplicates()
            print("CORAAL dedup df len ", len(df))
            full_df = df.merge(
                metadata,
                left_on=["basefile", "speaker"],
                right_on=["CORAAL.File", "CORAAL.Spkr"],
                how="left",
            )
            print("CORAAL full merged metadata len ", len(full_df))
            return full_df

        def find_snippet(snippets, basefile, start_time, end_time):
            start_time = round(start_time, 3)
            end_time = round(end_time, 3)
            match = snippets[
                (snippets.basefile == basefile)
                & (round(snippets.start_time, 3) == start_time)
                & (round(snippets.end_time, 3) == end_time)
            ]
            if len(match) == 0:
                print(
                    "Snippet not found at {} from {} to {}".format(
                        basefile, start_time, end_time
                    )
                )
            return match

        def segment_filename(basefilename, start_time, end_time, buffer):
            start_time = int((start_time - buffer) * 1000)
            end_time = int((end_time + buffer) * 1000)
            filename = "{}_{}_{}.wav".format(basefilename, start_time, end_time)
            return filename

        def create_coraal_snippets(transcripts):
            snippets = []

            for basefile in transcripts.basefile.unique():
                df = transcripts[transcripts.basefile == basefile][
                    [
                        "line",
                        "start_time",
                        "end_time",
                        "interviewee",
                        "content",
                        "Gender",
                        "Age",
                    ]
                ]
                backward_check = (
                    df["start_time"].values[1:] >= df["end_time"].values[:-1]
                )
                backward_check = np.insert(backward_check, 0, True)
                forward_check = (
                    df["end_time"].values[:-1] <= df["start_time"].values[1:]
                )
                forward_check = np.insert(forward_check, len(forward_check), True)
                df["use"] = (
                    backward_check
                    & forward_check
                    & df.interviewee
                    & ~df.content.str.contains("\[")
                    & ~df.content.str.contains("]")
                )

                values = df[["line", "use"]].values
                snippet = []
                for i in range(len(values)):
                    line, use = values[i]
                    if use:
                        snippet.append(line)
                    elif snippet:  # if shouldn't use this line, but snippet exists
                        snippets.append(snippet)
                        snippet = []
                if snippet:
                    snippets.append(snippet)

            basefiles = transcripts.basefile.values
            start_times = transcripts.start_time.values
            end_times = transcripts.end_time.values
            contents = transcripts.content.values
            gender = transcripts.Gender.values
            age = transcripts.Age.values
            rows = []
            for indices in snippets:
                rows.append(
                    {
                        "basefile": basefiles[indices[0]],
                        "start_time": start_times[indices[0]],
                        "end_time": end_times[indices[-1]],
                        "content": " ".join(contents[indices]),
                        "age": age[indices[0]],
                        "gender": gender[indices[0]],
                    }
                )
            snippets = pd.DataFrame(rows)[
                ["basefile", "start_time", "end_time", "content", "age", "gender"]
            ]
            snippets = snippets.sort_values(["basefile", "start_time"])
            snippets["duration"] = snippets.end_time - snippets.start_time
            snippets["segment_filename"] = [
                segment_filename(b, s, e, buffer=0)
                for b, s, e in snippets[["basefile", "start_time", "end_time"]].values
            ]
            return snippets

        coraal_transcripts = load_coraal_text(eval_dir)

        coraal_snippets = create_coraal_snippets(coraal_transcripts)

        # These snippets should exist, run these pre-filtering on duration
        assert (
            len(find_snippet(coraal_snippets, "DCB_se1_ag1_f_01_1", 8.9467, 12.4571))
            > 0
        )
        assert (
            len(find_snippet(coraal_snippets, "DCB_se1_ag1_f_01_1", 364.6292, 382.2063))
            > 0
        )
        assert (
            len(find_snippet(coraal_snippets, "DCB_se1_ag1_f_01_1", 17.0216, 19.5291))
            > 0
        )
        assert (
            len(find_snippet(coraal_snippets, "DCB_se1_ag1_f_01_1", 875.0084, 876.5177))
            > 0
        )
        assert (
            len(find_snippet(coraal_snippets, "DCB_se1_ag1_f_01_1", 885.9359, 886.3602))
            > 0
        )
        assert (
            len(find_snippet(coraal_snippets, "DCB_se1_ag1_f_01_1", 890.9707, 894.35))
            > 0
        )
        assert (
            len(find_snippet(coraal_snippets, "DCB_se1_ag1_f_01_1", 895.9076, 910.211))
            > 0
        )

        assert (
            len(
                coraal_snippets[
                    (coraal_snippets.content.str.contains("\["))
                    | (coraal_snippets.content.str.contains("\]"))
                ]
            )
            == 0
        )
        interviewees = {
            b: s
            for b, s in coraal_transcripts[coraal_transcripts.interviewee][
                ["basefile", "speaker"]
            ]
            .drop_duplicates()
            .values
        }

        for basefile, start_time, end_time in coraal_snippets[
            ["basefile", "start_time", "end_time"]
        ].values:
            xscript_speakers = coraal_transcripts[
                (coraal_transcripts.basefile == basefile)
                & (coraal_transcripts.start_time >= start_time)
                & (coraal_transcripts.end_time <= end_time)
            ].speaker.unique()
            if len(xscript_speakers) < 1:
                print(basefile, start_time, end_time)
                assert 0
            if not (
                len(xscript_speakers) == 1
                and xscript_speakers[0] == interviewees[basefile]
            ):
                print(basefile, start_time, end_time)
                assert 0

        min_duration = 5  # in seconds
        max_duration = 50  # in seconds

        coraal_snippets = coraal_snippets[
            (min_duration <= coraal_snippets.duration)
            & (coraal_snippets.duration <= max_duration)
        ]
        coraal_snippets.to_csv(
            os.path.join(eval_dir, "coraal_snippets.tsv"), sep="\t", index=False
        )
        
        def create_segment(src_file, dst_dir, segment_basename, start_time, end_time, buffer):
            segment_file = os.path.join(dst_dir, segment_basename)
            
            if not os.path.isfile(src_file):
                print("Error: Source file {} not found".format(src_file))
                return
            
            os.makedirs(dst_dir, exist_ok=True)
            audio = AudioSegment.from_wav(src_file)
            start_time = int((start_time-buffer)*1000)
            end_time = int((end_time+buffer)*1000)
            clip = audio[start_time:end_time]
            clip.export(segment_file, format="wav")
            return segment_file

        def parallel_create_segment(snippet):
            basefile, start_time, end_time, *_, segment_basename = snippet
            start_time = float(start_time)
            end_time = float(end_time)
            sub_folder = os.path.join(eval_dir, basefile.split('_')[0].lower())
            src_file = os.path.join(sub_folder, 'audio', basefile + '.wav')
            create_segment(src_file, os.path.join(sub_folder, 'segments'), segment_basename, start_time, end_time, 0)
            return segment_basename

        coraal_snippets = coraal_snippets.values.tolist()
        with multiprocessing.Pool() as pool:
            res = list(tqdm(pool.imap_unordered(parallel_create_segment, coraal_snippets), total=len(coraal_snippets)))
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

if __name__ == "__main__":
    Fire(get_eval_set)
