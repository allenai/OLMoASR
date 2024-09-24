import multiprocessing
from tqdm import tqdm
import tarfile
import os
from typing import Optional, List
from fire import Fire

unzip_files_path = "/mmfs1/gscratch/efml/hvn2002/ow_440K" 

def unzip_tar(tar_file: str) -> None:
    output_dir = tar_file.split("/")[-1].split(".")[0]
    with tarfile.open(tar_file, 'r:gz') as file:
        file.extractall(path=os.path.join(unzip_files_path, output_dir))



def main(tar_files: Optional[str]):
    tar_files_path = "/mmfs1/gscratch/efml/hvn2002/ow_440K_tar"
    os.makedirs(unzip_files_path, exist_ok=True)
    if tar_files:
        tar_files_list = tar_files.split(",")
        tar_files_paths = [os.path.join(tar_files_path, local) for local in tar_files_list]
    else:
        tar_files_paths = [os.path.join(tar_files_path, local) for local in os.listdir(tar_files_path)]

    with multiprocessing.Pool() as pool:
        out = list(
            tqdm(
                pool.imap_unordered(
                    unzip_tar,
                    tar_files_paths,
                ),
                total=len(tar_files_paths),
            )
        )

if __name__ == "__main__":
    Fire(main)