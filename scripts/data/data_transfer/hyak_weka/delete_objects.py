import boto3
import multiprocessing
from more_itertools import bucket
from tqdm import tqdm
from itertools import repeat

s3 = boto3.client(
    "s3",
    endpoint_url="https://b2bcf985082a37eaf385c532ee37928d.r2.cloudflarestorage.com",
    aws_access_key_id="f99d33cc371208673054c6d7500d802d",
    aws_secret_access_key="4dc41563097403fd225f47e6486b89dca0973656612b657ec5e255221ab36cfa",
)

bucket_name = "open-whisper"


def delete_object(bucket_name, key):
    try:
        s3.delete_object(Bucket=bucket_name, Key=key)
    except:
        with open("logs/data/download/fail_delete_obj.txt", "a") as f:
            f.write(f"{key}\n")


def parallel_delete(args):
    return delete_object(*args)


def abort_upload(bucket_name, upload_id, key):
    try:
        s3.abort_multipart_upload(Bucket=bucket_name, UploadId=upload_id, Key=key)
    except:
        with open("logs/data/download/fail_abort.txt", "a") as f:
            f.write(f"{key}\n")


def parallel_abort(args):
    return abort_upload(*args)


def delete_objects(bucket_name):
    # List all objects in the bucket
    while True:
        object_keys = [
            obj["Key"] for obj in s3.list_objects_v2(Bucket=bucket_name)["Contents"]
        ]
        if len(object_keys) == 0:
            break
        print(object_keys[:10])
        print(f"Deleting {len(object_keys)} objects in bucket {bucket_name}")

        with multiprocessing.Pool() as pool:
            result = list(
                tqdm(
                    pool.imap_unordered(
                        parallel_delete, zip(repeat(bucket_name), object_keys)
                    ),
                    total=len(object_keys),
                )
            )
        print(f"All objects in bucket {bucket_name} deleted successfully")


def abort_uploads(bucket_name):
    while True:
        bucket_uid_key = [
            (bucket_name, d["UploadId"], d["Key"])
            for d in s3.list_multipart_uploads(Bucket=bucket_name)["Uploads"]
        ]

        with multiprocessing.Pool() as pool:
            result = list(
                tqdm(
                    pool.imap_unordered(
                        parallel_abort, bucket_uid_key
                    ),
                    total=len(bucket_uid_key),
                )
            )
        print(f"All multipart uploads are aborted successfully")


def delete_bucket(bucket_name):
    s3.delete_bucket(Bucket=bucket_name)
    print(f"Bucket {bucket_name} deleted successfully")


if __name__ == "__main__":
    abort_uploads(bucket_name)
    # delete_objects(bucket_name)
    delete_bucket(bucket_name)
