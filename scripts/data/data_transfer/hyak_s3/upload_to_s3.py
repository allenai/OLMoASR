from moto3.s3_manager import S3Manager
import os
from fire import Fire

s3m = S3Manager("mattd-public/whisper")

def main(group_id: str) -> None:
    """Upload a tar file to S3

    Args:
        group_id: Folder and name of tar file to upload
    """
    s3m.upload_file(f"{group_id}.tar.gz", f"{group_id}.tar.gz")
    
    os.remove(f"{group_id}.tar.gz")

if __name__ == "__main__":
    Fire(main)