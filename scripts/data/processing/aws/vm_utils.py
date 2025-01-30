import boto3
import requests


def name_instances(instance_ids, name_pattern):
    # Initialize EC2 client
    ec2_client = boto3.client("ec2")

    for idx, instance_id in enumerate(instance_ids, start=1):
        ec2_client.create_tags(
            Resources=[instance_id],
            Tags=[{"Key": "Name", "Value": f"{name_pattern}-{idx}"}],
        )
        print(f"Instance {instance_id} named as {name_pattern}-{idx}")
    
    return name_pattern

def launch_instances(
    name_pattern: str = "ow",
    image_id: str = "ami-09115b7bffbe3c5e4",
    instance_type: str = "i4i.12xlarge",  # e.g., "t2.micro"
    key_name: str = "huongn",  # e.g., "my-key-pair"
    instance_count=1,  # Number of instances to launch
    script=None,  # User data script
    max_price: str = "2",  # Max price you're willing to pay for spot instances
):
    # Create an EC2 client
    ec2_client = boto3.client("ec2")

    # Launch instances
    response = ec2_client.run_instances(
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {"DeleteOnTermination": True, 'SnapshotId': 'snap-032239a8aa7c2f100'}}],
        ImageId=image_id,
        InstanceType=instance_type,
        KeyName=key_name,
        MinCount=instance_count,
        MaxCount=instance_count,
        UserData=script,
        InstanceMarketOptions={
            "MarketType": "spot",
            "SpotOptions": {
                "MaxPrice": max_price,  # Optional: Specify the max price for the spot instance
                "SpotInstanceType": "one-time",  # "one-time" or "persistent"
                "InstanceInterruptionBehavior": "terminate"  # Options: "terminate", "stop", "hibernate"
            }
        },
    )

    # Extract instance IDs from the response
    instance_ids = [instance["InstanceId"] for instance in response["Instances"]]
    name_pattern = name_instances(instance_ids, name_pattern=name_pattern)
    return name_pattern, instance_ids, response

# def launch_instances(
#     name_pattern: str = "ow",
#     image_id: str = "ami-09115b7bffbe3c5e4",
#     instance_type: str = "i4i.12xlarge",  # e.g., "t2.micro"
#     key_name: str = "huongn",  # e.g., "my-key-pair"
#     instance_count=1,  # Number of instances to launch
#     script=None,  # User data script
# ):
#     # Create an EC2 client
#     ec2_client = boto3.client("ec2")

#     # Launch instances
#     response = ec2_client.run_instances(
#         BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {"DeleteOnTermination": True, 'SnapshotId': 'snap-032239a8aa7c2f100'}}],
#         ImageId=image_id,
#         InstanceType=instance_type,
#         KeyName=key_name,
#         MinCount=instance_count,
#         MaxCount=instance_count,
#         UserData=script,
#     )

#     # Extract instance IDs from the response
#     instance_ids = [instance["InstanceId"] for instance in response["Instances"]]
#     name_pattern = name_instances(instance_ids, name_pattern=name_pattern)
#     return name_pattern, instance_ids, response


def terminate_instance():
    try:
        # Get the instance ID and region from the instance metadata
        instance_id = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-id"
        ).text
        region = requests.get(
            "http://169.254.169.254/latest/meta-data/placement/region"
        ).text

        print(f"Instance ID: {instance_id}")
        print(f"Region: {region}")

        # Create an EC2 client
        ec2 = boto3.client("ec2", region_name=region)

        # Terminate the instance
        response = ec2.terminate_instances(InstanceIds=[instance_id])
        print(f"Terminate response: {response}")

        print(f"Instance {instance_id} has been terminated.")
    except Exception as e:
        print(f"Error: {e}")


def terminate_instances(instance_ids):
    ec2_client = boto3.client("ec2")
    ec2_client.terminate_instances(InstanceIds=instance_ids)
    print(f"Terminated instances: {instance_ids}")
