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
    service_account: str,
    key_file: str,
    log_file: str,
):
    os.makedirs(local_dir, exist_ok=True)
    start_idx = int(os.getenv("BEAKER_REPLICA_RANK")) * batch_size
    end_idx = ((start_idx * batch_size) + batch_size) - 1
    cmd = [
        "gcloud",
        "auth",
        "activate-service-account",
        f"'{service_account}'",
        f"--key-file='{key_file}'",
        "&&",
        "gsutil",
        "-m",
        "cp",
        "-L",
        f"{log_file}",
        "-r",
        f"gs://huongn-openwhisper/[{start_idx:04}-{end_idx:04}].tar.gz",
        f"{local_dir}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True)
        logger.info(result.stdout)
        logger.info(result.stderr)
    except Exception as e:
        logger.info(result.stderr)
        logger.error(e)


if __name__ == "__main__":
    Fire(download_files)