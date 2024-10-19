import subprocess
import os
import logging
from fire import Fire

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.FileHandler("download.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def download_files(
    batch_size: int,
    local_dir: str,
    bucket_name: str,
    bucket_prefix: str,
    service_account: str,
    key_file: str,
    log_file: str,
):
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    start_idx = int(os.getenv("BEAKER_REPLICA_RANK")) * batch_size
    end_idx = start_idx + batch_size

    for file_idx in range(start_idx, end_idx):
        cmd = (
            f"gcloud auth activate-service-account '{service_account}' --key-file='{key_file}' && gsutil -m cp -L {log_file} -r gs://{bucket_name}/{bucket_prefix}/{file_idx:04}.tar.gz {local_dir}",
        )
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True)
            logger.info(result.stdout)
            logger.info(result.stderr)
        except Exception as e:
            logger.info(result.stderr)
            logger.error(e)


if __name__ == "__main__":
    Fire(download_files)
