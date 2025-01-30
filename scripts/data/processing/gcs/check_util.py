from google.cloud import monitoring_v3, compute_v1
from google.protobuf import timestamp_pb2
import time
import re
from fire import Fire

# Set your project ID, zone, and other parameters
PROJECT_ID = "oe-training"
CPU_THRESHOLD = 1  # CPU utilization threshold in percentage
MONITORING_INTERVAL = 120  # Check interval in seconds

def get_cpu_utilization(project_id, instance_id, zone):
    """Retrieve the CPU utilization of a specified VM instance."""
    client = monitoring_v3.MetricServiceClient()
    interval = monitoring_v3.TimeInterval()

    # Set end time to the current time
    interval_end = timestamp_pb2.Timestamp()
    interval_end.GetCurrentTime()
    interval.end_time = interval_end

    # Set start time to 5 minutes before the current time
    interval_start = timestamp_pb2.Timestamp()
    interval_start.FromSeconds(interval_end.seconds - 300)
    interval.start_time = interval_start

    resource_filter = (
        f'resource.type="gce_instance" AND '
        f'resource.labels.instance_id="{instance_id}" AND '
        f'resource.labels.zone="{zone}"'
    )

    # CPU utilization metric
    results = client.list_time_series(
        request={
            "name": f"projects/{project_id}",
            "filter": f'metric.type="compute.googleapis.com/instance/cpu/utilization" AND {resource_filter}',
            "interval": interval,
            "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        }
    )

    # Average CPU utilization over the period
    cpu_usage = 0
    points_count = 0
    for result in results:
        for point in result.points:
            cpu_usage += point.value.double_value
            points_count += 1

    return (cpu_usage / points_count) * 100 if points_count > 0 else 0

def reset_instance(project_id, zone, instance_name):
    """Reset the specified VM instance."""
    client = compute_v1.InstancesClient()
    try:
        operation = client.reset(project=project_id, zone=zone, instance=instance_name)
        print(f"Resetting instance {instance_name}. Operation ID: {operation.name}")
    except Exception as e:
        print(f"Error resetting instance {instance_name}: {e}")


def shutdown_vm(project, zone, instance_name):
    client = compute_v1.InstancesClient()
    # Request to stop the instance
    operation = client.delete(project=project, zone=zone, instance=instance_name)
    # Wait for the operation to complete
    operation.result()
    print(f"Deleting instance {instance_name}. Operation ID: {operation.name}")

def get_instances_by_base_name(project_id, zone, base_name):
    """List all instances matching the base name in a specified zone."""
    client = compute_v1.InstancesClient()
    instances = client.list(project=project_id, zone=zone)
    return [instance for instance in instances if re.match(f"^{base_name}", instance.name)]

def main(zone, base_name):
    while True:
        print("Starting monitoring loop...")
        instances = get_instances_by_base_name(PROJECT_ID, zone, base_name)
        print(f"Found {len(instances)} instances with base name '{base_name}'.")

        for instance in instances:
            cpu_utilization = get_cpu_utilization(PROJECT_ID, instance.id, zone)
            print(f"Instance {instance.name} - Current CPU utilization: {cpu_utilization:.2f}%")

            if cpu_utilization < CPU_THRESHOLD:
                print(f"Instance {instance.name} - CPU utilization ({cpu_utilization:.2f}%) below threshold. Resetting.")
                reset_instance(PROJECT_ID, zone, instance.name)
            else:
                print(f"Instance {instance.name} - CPU utilization ({cpu_utilization:.2f}%) is within limits.")

        print(f"Waiting for {MONITORING_INTERVAL} seconds...")
        time.sleep(MONITORING_INTERVAL)

if __name__ == "__main__":
    Fire(main)