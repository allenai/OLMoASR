import multiprocessing
from tabnanny import check
from tqdm import tqdm
import shutil
import glob
import os
from fire import Fire

OW_440K = "/mmfs1/gscratch/efml/hvn2002/ow_440K"

def check_dir(video_id_dir: str):
    if len(os.listdir(video_id_dir)) != 2:
        shutil.rmtree(video_id_dir)
        return video_id_dir.split("/")[-1]
    elif len(glob.glob(video_id_dir + "/*.m4a.part")) > 0 :
        shutil.rmtree(video_id_dir)
        return video_id_dir.split("/")[-1]
    return None

def check_data_shard(data_shard_idx: int):
    
    data_shard_path = os.path.join(OW_440K, f"{data_shard_idx:08d}")
    video_id_dirs = glob.glob(data_shard_path + "/*")

    with multiprocessing.Pool() as pool:
        out = list(
            tqdm(
                pool.imap_unordered(check_dir, video_id_dirs),
                total=len(video_id_dirs),
            )
        )

    with open("logs/data/preprocess/uneven_video_id.txt", "a") as f:
        for video_id in out:
            if video_id is not None:
                f.write(f"{video_id}\n")

if __name__ == "__main__":
    Fire(check_data_shard)