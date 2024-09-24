import tarfile
import numpy as np
import io
import multiprocessing
from tqdm import tqdm
import time
import os
import glob
from fire import Fire
from tempfile import NamedTemporaryFile
import shutil

CUSTOM_TEMP_DIR = "/mmfs1/gscratch/efml/hvn2002/temp_dir"

def convert_npy_precision_in_tar(tar_path: str):
    # Open the tar file in read mode
    start = time.time()
    npy_files = []
    srt_files = []
    
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        npy_members = [m for m in members if m.name.endswith('.npy')]
        srt_members = [m for m in members if m.name.endswith('.srt')]

        for i in range(len(npy_members)):
            # srt file
            f = tar.extractfile(srt_members[i])
            srt_data = f.read()
            new_srt_data = io.BytesIO(srt_data)
            srt_files.append((srt_members[i].name, new_srt_data))

            # npy file
            f = tar.extractfile(npy_members[i])
            npy_data = f.read()
            npy_array = np.load(io.BytesIO(npy_data))
            
            # Convert to int16 if it's float32
            if npy_array.dtype == np.float32:
                npy_array = npy_array * 32768.0
                npy_array = npy_array.astype(np.int16)
            else:
                print(f"{tar_path} is already in int16 format")
                return None
            
            # Save the updated array to a BytesIO object
            new_npy_data = io.BytesIO()
            np.save(new_npy_data, npy_array)
            new_npy_data.seek(0)
            npy_files.append((npy_members[i].name, new_npy_data))
    
    print(f"{tar_path} took {time.time() - start} seconds to update npy files")

    # Write the updated tar file
    with NamedTemporaryFile(delete=False, suffix=".tar", dir=CUSTOM_TEMP_DIR) as temp_tar:
        temp_tar_path = temp_tar.name
        
        with tarfile.open(temp_tar_path, 'w') as tar:
            for i in range(len(npy_files)):
                name, data = npy_files[i]
                info = tarfile.TarInfo(name=name)
                info.size = len(data.getvalue())
                data.seek(0)
                tar.addfile(info, fileobj=data)

                name, data = srt_files[i]
                info = tarfile.TarInfo(name=name)
                info.size = len(data.getvalue())
                data.seek(0)
                tar.addfile(info, fileobj=data)
    
    shutil.move(temp_tar_path, tar_path)
    
    print(f"{tar_path} took {time.time() - start} seconds to finish")

if __name__ == '__main__':
    Fire(convert_npy_precision_in_tar)