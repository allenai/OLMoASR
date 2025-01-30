from moto3.queue_manager import QueueManager
import json
from fire import Fire
from google.cloud import storage
from typing import Optional

def main(queue_name: str, batch_file: Optional[str] = None):
    qm = QueueManager(queue_name)   
    # batches = [{"tarFile": f"{i:08}.tar.gz"} for i in range(16000, 26000)]
    # qm.purge()
    with open(batch_file, "r") as f:
        items = [json.loads(line.strip()) for line in f]
    # with open(batch_file, "r") as f:
    #     dicts = [json.loads(line.strip()) for line in f]
    
    # items = [{"tarFile": f"{d['batchIdx']:08}_{d['language']}.tar.gz"} for d in dicts if d['batchIdx'] not in {2436, 2437, 2438, 2439}]
    
    # client = storage.Client()
    # bucket = client.get_bucket("ow-download-ml")
    # items = [{"tarFile": blob.name.split("/")[-1]} for blob in bucket.list_blobs(prefix="ow_ml_full_rnd2")]
    # print(len(items))
    # print(items[:10])
    for item in items[1000:]:
        qm.upload([item])

if __name__ == '__main__':
    # main("ow-seg")
    Fire(main)
    # main("ow-seg", "logs/data/download/9M_ml/subsampled_batches.jsonl")