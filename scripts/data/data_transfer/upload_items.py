from moto3.queue_manager import QueueManager
from moto3.s3_manager import S3Manager

s3m = S3Manager("mattd-public/whisper")
qm = QueueManager("whisper-downloading")

# make the items
items = []
with open("logs/data/download/sampled_en.txt", "r") as f:
    id_lang = [line.strip().split("\t")[:2] for line in f]

uploaded = []
for f in s3m.list_all_files():
    if "tar.gz" in f:
        uploaded.append(int(f.split(".")[0]))

for i in range(0, len(id_lang), 1000):
    if i // 1000 in uploaded:
        pass
    else:
        item = {"id_lang": id_lang[i : i + 1000], "group_id": f"{((i // 1000)):08}"}
        items.append(item)
    if i % 3000 == 0 and i != 0:
        qm.upload(items)
        items = []
        
