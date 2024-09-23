import subprocess
import multiprocessing
from tqdm import tqdm
import glob
from fire import Fire
import json

def upload_to_r2(file_path: str):
    command = ["s5cmd", "cp", "--concurrency", "10", file_path, "s3://open-whisper/ow_440K_wds/"]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {"file_path": file_path, "stdout": result.stdout.decode("utf-8"), "stderr": result.stderr.decode("utf-8")}
    except:
        return {"file_path": file_path, "stdout": result.stdout.decode("utf-8"), "stderr": result.stderr.decode("utf-8")}
        

def bulk_upload(tars_path: str, job_index: int, batch_size: int, log_path: str):
    all_tar_files = glob.glob(tars_path + "/*.tar")
    tar_files_list = all_tar_files[job_index * batch_size: (job_index + 1) * batch_size]
    
    print(f"Uploading from {job_index * batch_size} to {(job_index + 1) * batch_size}")
    
    with multiprocessing.Pool(multiprocessing.cpu_count() * 7) as pool:
        result = list(tqdm(pool.imap_unordered(upload_to_r2, tar_files_list), total=len(tar_files_list)))
    with open(log_path, "a") as f:
        for res in result:
            f.write(json.dumps(res) + "\n")

    return "Done"

if __name__ == "__main__":
    Fire({"bulk_upload": bulk_upload, "upload_to_r2": upload_to_r2})