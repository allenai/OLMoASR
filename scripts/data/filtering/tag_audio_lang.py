from speechbrain.inference.classifiers import EncoderClassifier
from collections import defaultdict
from tqdm import tqdm
from itertools import chain
import os
import glob
from fire import Fire
import numpy as np
import json
import gzip
import torch
import multiprocessing
from typing import List, Dict
from whisper import audio
from torch.utils.data import Dataset, DataLoader


class SamplesDictsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_file = self.data[idx]["audio_file"].replace("ow_full", "ow_seg")
        video_id = self.data[idx]["id"]
        content = self.data[idx]["content"]
        audio_arr = self.load_audio(audio_file)
        return video_id, audio_arr, content

    def load_audio(self, audio_file):
        audio_arr = np.load(audio_file).astype(np.float32) / 32768.0
        audio_arr = audio.pad_or_trim(audio_arr)
        return audio_arr


def open_file(file_path) -> List[Dict]:
    with gzip.open(file_path, "rt") as f:
        data = [json.loads(line.strip()) for line in f]
    return data


def modify_sample_dict(sample_dict, lang_id):
    sample_dict["audio_lang"] = lang_id
    return sample_dict


def main(
    source_dir: str,
    output_dir: str,
    job_batch_size: int,
    batch_size: int,
    num_workers: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    start_shard_idx = int(os.getenv("START_SHARD_IDX"))
    job_idx = int(os.getenv("BEAKER_REPLICA_RANK"))
    job_start_shard_idx = start_shard_idx + (job_idx * job_batch_size)
    job_end_shard_idx = start_shard_idx + ((job_idx + 1) * job_batch_size)
    data_shard_paths = sorted(glob.glob(source_dir + "/*"))[
        job_start_shard_idx : job_end_shard_idx
    ]
    print(f"{len(data_shard_paths)=}")
    print(f"{data_shard_paths[:5]=}")
    print(f"{data_shard_paths[-5:]=}")

    with multiprocessing.Pool() as pool:
        data = list(
            chain(
                *tqdm(
                    pool.imap_unordered(open_file, data_shard_paths),
                    total=len(data_shard_paths),
                )
            )
        )

    print(f"{len(data)=}")

    device = torch.device("cuda")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-voxlingua107-ecapa",
        savedir="tmp",
        run_opts={"device": "cuda"},
    )

    dataset = SamplesDictsDataset(data)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
    )

    pred_lang_ids = []
    for _, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        video_ids, audio_arr = batch
        audio_arr = audio_arr.to(device)
        results = model.classify_batch(audio_arr)
        pred_lang_ids.extend(
            [(video_ids[i], res.split(": ")[0]) for i, res in enumerate(results[3])]
        )

    ids_to_lang_dist = defaultdict(list)
    for video_id, lang_id in pred_lang_ids:
        ids_to_lang_dist[video_id].append(lang_id)

    ids_to_lang = {
        video_id: max(langs, key=langs.count)
        for video_id, langs in ids_to_lang_dist.items()
    }

    print(f"{len(pred_lang_ids)} samples tagged")
    print(f"{len(ids_to_lang)} unique video ids")

    output_path = os.path.join(
        output_dir, f"ids_to_lang_{job_start_shard_idx:08}_{job_end_shard_idx:08}.json"
    )

    with gzip.open(output_path, "wt") as f:
        json.dump(ids_to_lang, f, indent=2)

    # new_data = [modify_sample_dict(sample_dict, lang_id) for sample_dict, lang_id in zip(data, pred_lang_ids)]

    # output_path = os.path.join(output_dir, os.path.basename(data_shard_path))

    # with gzip.open(output_path, "wt") as f:
    #     for line in new_data:
    #         f.write(json.dumps(line) + "\n")

    print(f"Saved tagged data to {output_path}")


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn', force=True)
    Fire(main)
