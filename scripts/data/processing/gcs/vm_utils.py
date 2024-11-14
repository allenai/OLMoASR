# from action-atlas
import subprocess
import logging
from fire import Fire
import re
from google.cloud import compute_v1
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - line %(lineno)d - %(message)s",
    handlers=[logging.FileHandler("./vm_utils.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def create_instance(
    project_id: str = "oe-training",
    service_account: str = "349753783513-compute@developer.gserviceaccount.com",
    startup_script_path: str = "debug_startup_script.sh",
    instance_name: Optional[str] = None,
    zone: str = "us-central1-c",
    machine_type: str = "n2d-highcpu-96",
    disk_size: int = 100,
):
    if not instance_name:
        instance_name = f"ow-{np.random.randint(low=0, high=100)}"

    command = [
        "gcloud",
        "compute",
        "instances",
        "create",
        instance_name,
        "--network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default",
        "--provisioning-model=STANDARD",
        f"--service-account={service_account}",
        f"--zone={zone}",
        f"--project={project_id}",
        f"--machine-type={machine_type}",
        f"--metadata-from-file=startup-script={startup_script_path}",
        "--scopes=cloud-platform",
        "--tags=http-server,https-server",
        f"--create-disk=auto-delete=yes,boot=yes,device-name=instance-20241016-003225,image=projects/debian-cloud/global/images/debian-12-bookworm-v20241009,mode=rw,size={disk_size},type=pd-balanced",
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Error: Command failed with {result.stderr}")

    return result


def bulk_create_spot_instances(
    iteration: Optional[int],
    project_id: str = "oe-training",
    service_account: str = "349753783513-compute@developer.gserviceaccount.com",
    startup_script_path: str = "startup_script.sh",
    count: int = 1,
    zone: str = "us-central1-c",
    machine_type: str = "n2d-highcpu-96",
    termination_action: str = "DELETE",
    base_name: str = "ow-download",
    disk_size: int = 100,
):
    # Construct the gcloud command
    name_pattern = f"{base_name}-{iteration}-###" if iteration else f"{base_name}-###"
    command = [
        "gcloud",
        "compute",
        "instances",
        "bulk",
        "create",
        f"--name-pattern={name_pattern}",
        f"--zone={zone}",
        f"--project={project_id}",
        f"--count={count}",
        f"--metadata-from-file=startup-script={startup_script_path}",
        "--provisioning-model=SPOT",
        f"--machine-type={machine_type}",
        f"--instance-termination-action={termination_action}",
        "--no-restart-on-failure",
        f"--service-account={service_account}",
        "--scopes=cloud-platform",
        f"--create-disk=auto-delete=yes,boot=yes,device-name=instance-20241016-003225,image=projects/debian-cloud/global/images/debian-12-bookworm-v20241009,mode=rw,size={disk_size},type=pd-balanced",
    ]

    # Execute the command and capture the output
    result = subprocess.run(command, capture_output=True, text=True)
    # Check if there were any errors
    if result.returncode != 0:
        logger.error(f"Error: Command failed with {result.returncode}")

    return result


def list_instances(project_id, zone, name_pattern):
    compute_client = compute_v1.InstancesClient()
    request = compute_v1.ListInstancesRequest(project=project_id, zone=zone)
    instances = compute_client.list(request=request)
    return [
        instance.name for instance in instances if re.match(name_pattern, instance.name)
    ]


def delete_instance(project_id, zone, instance_name):
    compute_client = compute_v1.InstancesClient()
    operation = compute_client.delete(
        project=project_id, zone=zone, instance=instance_name
    )
    print(f"Instance {instance_name} deleted successfully")


def bulk_delete_instances_by_pattern(project_id, zone, name_pattern):
    instance_names = list_instances(project_id, zone, name_pattern)

    if not instance_names:
        logger.info(f"No instances found matching the pattern '{name_pattern}'")
        return

    logger.info(f"Deleting the following instances: {', '.join(instance_names)}")

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(delete_instance, project_id, zone, name)
            for name in instance_names
        ]
        for future in futures:
            future.result()

    logger.info("Deletion process completed.")


if __name__ == "__main__":
    Fire(
        {
            "bulk_create": bulk_create_spot_instances,
            "bulk_delete": bulk_delete_instances_by_pattern,
            "list_instances": list_instances,
            "delete_instance": delete_instance,
            "create_instance": create_instance,
        }
    )
