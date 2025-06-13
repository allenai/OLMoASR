import glob
import os 
from typing import Tuple, Union, Dict, Any, Literal, Optional, List
import numpy as np
import io
from collections import defaultdict
from smart_open import open
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import Process, Manager, Lock, Queue
import json 

from functools import partial
import yaml
import time
import argparse
import tabulate
import random
import tempfile


"""
Does some hacky parallel reservoir sampling thing.
To avoid nonsense threadsafety concerns, here's how this will work:
if we want to run on k threads and collect N samples total, we'll:
    1. partition our files into k groups
    2. Run reservoir sampling to collect N / k elements from each group

And then once are al done, we'll merge them, and print out and save the percentiles

*Note, this isn't perfect reservoir sampling, but it's good enough. 
If you want perfect reservoir sampling, set num_cpus == 1 ;)
"""



# ======================================================
# =                       UTILITIES                    =
# ======================================================

def run_imap_multiprocessing(func, argument_list, num_processes):
    # Stolen from https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
    pool = Pool(processes=num_processes)

    result_list = []
    for result in pool.imap(func=func, iterable=argument_list):
        result_list.append(result)

    return result_list



class SharedCounter:
    def __init__(self, init_val=0):
        self.val = Manager().Value('i', init_val)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1
        return self.val.value


# ======================================================
# =                     THREAD LOGIC                   =
# ======================================================


def process_chunk(chunk, key, chunk_reservoir_size, counter, result_queue):

    temp = tempfile.NamedTemporaryFile(delete=False)
    docs_seen = 0
    mini_reservoir = []
    for file in chunk:
        lines = [json.loads(_) for _ in open(file, 'rb').read().splitlines()]
        for line in lines:
            value = line.get(key)

            if len(mini_reservoir) < chunk_reservoir_size:
                mini_reservoir.append(value)
            else:
                j = random.randint(0, docs_seen)
                if j < chunk_reservoir_size:
                    mini_reservoir[j] = value
            docs_seen += 1
        counter.increment()


    temp.write(json.dumps(mini_reservoir).encode('utf-8'))
    temp.flush()
    temp.close()
    print("FINISHED CHUNK")
    result_queue.put(temp.name)


# ======================================================
# =                     MAIN STUFF                     =
# ======================================================



def main(input_dir, key, output_loc, reservoir_size=1_000_000, num_cpus=None):
    start_time = time.time()
    if num_cpus == None:
        num_cpus = os.cpu_count()

    files = glob.glob(os.path.join(input_dir, '**/*.jsonl*'), recursive=True)
    chunks = [[] for _ in range(num_cpus)]
    for i, el in enumerate(files):
        chunks[i % num_cpus].append(el)


    chunk_reservoir_size = round(reservoir_size / num_cpus)
    counter = SharedCounter()

    print("Starting reservoir sampling for key %s | %s files | Res size: %s" % (key,  len(files), reservoir_size))

    # Claude's multiprocessing incantations
    pbar = tqdm(total=len(files), position=0, leave=True)
    processes = []
    
    result_queue = Queue()
    for _ in range(num_cpus):
        p = Process(target=process_chunk, args=(chunks[_], key, chunk_reservoir_size, counter, result_queue))
        processes.append(p)
        p.start()

    last_count = 0


    while any(p.is_alive() for p in processes):
        current_count = counter.val.value
        if current_count > last_count:
            pbar.update(current_count - last_count)
            last_count = current_count
        time.sleep(0.1)

    pbar.close()
    for p in processes:
        p.join()


    # Collect results from processes 
    reservoir = []


    while not result_queue.empty():
        temp_name = result_queue.get()
        with open(temp_name, 'rb') as f:
            reservoir.extend(json.loads(f.read()))
        os.unlink(temp_name)


    reservoir.sort()
    percentiles = [reservoir[round(i * len(reservoir) / 100)] for i in range(100)] + [reservoir[-1]]

    print(tabulate.tabulate(list(enumerate(percentiles)), headers=['percentile', 'value']))

    # and then print out result
    if output_loc:
        if os.path.dirname(output_loc):
            os.makedirs(os.path.dirname(output_loc), exist_ok=True)
        percentiles = json.dumps(percentiles)
        with open(output_loc, 'w') as f:
            f.write(percentiles)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-loc', type=str, default=None)
    parser.add_argument('--key', type=str, required=True)
    parser.add_argument('--reservoir-size', type=int, default=1_000_000)
    parser.add_argument('--num-cpus', type=int, required=False)
    args = parser.parse_args()

    main(args.input_dir, args.keys, args.output_loc, args.reservoir_size, args.num_cpus)

