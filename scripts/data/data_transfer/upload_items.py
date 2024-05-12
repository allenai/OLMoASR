from moto3.queue_manager import QueueManager

qm = QueueManager("whisper-downloading")

# make the items
items = []
with open("logs/data/download/sampled_en.txt", "r") as f:
    id_lang = [line.strip().split("\t")[:2] for line in f]

for i in range(0, len(id_lang), 1000):
    if i // 1000 in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 21, 36]:
        pass
    else:
        item = {"id_lang": id_lang[i : i + 1000], "group_id": f"{((i // 1000)):08}"}
        items.append(item)
    if i % 3000 == 0 and i != 0:
        qm.upload(items)
        items = []
        
