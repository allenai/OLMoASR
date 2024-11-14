from moto3.queue_manager import QueueManager
import json
from fire import Fire

def main(queue_name: str, batch_file):
    qm = QueueManager(queue_name)
    batches = [{"tarFile": f"{i:08}.tar.gz"} for i in range(0, 16000)]
    # qm.purge()
    # with open(batch_file, "r") as f:
    #     items = [json.loads(line.strip()) for line in f]

    for item in batches[8000:]:
        qm.upload([item])

if __name__ == '__main__':
    # Fire(main)
    main("ow-seg", None)