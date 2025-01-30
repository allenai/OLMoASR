import json
from tqdm import tqdm
import multiprocessing
from itertools import chain
import gzip

def get_id_to_shard(id_langs_batch):
    batch_idx = id_langs_batch[1]
    id_langs = id_langs_batch[0]
    return [(id_lang[0], batch_idx) for id_lang in id_langs]

if __name__ == "__main__":
    with open("logs/data/download/6M_en/subsampled_batches_machgen.jsonl", "r") as f:
        batches_1 = [json.loads(line.strip()) for line in f]

    with open("logs/data/download/4M_en/subsampled_batches_machgen.jsonl", "r") as f:
        batches_2 = [json.loads(line.strip()) for line in f]

    batches = batches_1 + batches_2

    id_langs_batch = [[batch["videoIds"], batch["batchIdx"]] for batch in batches]
    print(f"{len(id_langs_batch)=}")
    print(f"{id_langs_batch[0]=}")

    with multiprocessing.Pool() as pool:
        id_to_shard = list(
            chain(
                *tqdm(
                    pool.imap_unordered(get_id_to_shard, id_langs_batch),
                    total=len(id_langs_batch),
                )
            )
        )
    print(f"{len(id_to_shard)=}")
    print(f"{id_to_shard[0]=}")

    id_to_shard_dict = {id_lang: batch_idx for id_lang, batch_idx in id_to_shard}
    print(f"{len(id_to_shard_dict)=}")
    
    with gzip.open("logs/data/download/id_to_machgen_shard.json.gz", "wt") as f:
        json.dump(id_to_shard_dict, f, indent=1)

    # with open("logs/data/download/id_to_machgen_shard.json", "w") as f:
    #     json.dump(id_to_shard_dict, f, indent=1)
