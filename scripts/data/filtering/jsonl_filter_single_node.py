import glob
import os 
from typing import Tuple, Union, Dict, Any, Literal, Optional
from open_whisper.utils import TranscriptReader
import numpy as np
import io
from collections import defaultdict
import json 
import gzip
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import yaml
import time

"""
Single node ec2 data filtering pipeline. 
Just works local2local for now.

Usage will be like:
python jsonl_filter_single_node.py --config filter_config.yaml --input-dir <path/to/local/inputs> --output-dir <path/to/local/outputs>

(filter config setup TBD)

Will take a folder  of jsonl.gz's in and make a folder of jsonl.gz's out
Each line in a jsonl.gz is a json-dict like: 

{'audio_file': '/weka/oe-data-default/huongn/ow_full/00000548/uVC2h48KRos/uVC2h48KRos.m4a',
 'content': 'DUMMY SHORT CONTENT',
 'length': 303.6649375,
 'subtitle_file': '/weka/oe-data-default/huongn/ow_full/00000548/uVC2h48KRos/uVC2h48KRos.en-nP7-2PuUl7o.srt'}


[and only the 'content' key will be modified, but some lines may be deleted!]
-------------------


"""


# =============================================================
# =                          UTILITIES                        =
# =============================================================

def run_imap_multiprocessing(func, argument_list, num_processes):
    # Stolen from https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
    pool = Pool(processes=num_processes)

    result_list_tqdm = []
    for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list)):
        result_list_tqdm.append(result)

    return result_list_tqdm


def parse_config(config_path):
    config_dict = yaml.safe_load(open(config_path, 'r'))
    return config_dict




# =============================================================
# =                         FILTERING ATOMS                   =
# =============================================================

def _filter_stub(content: str, **kwargs):
    """ Basic filtering stub. Always takes in a content and some kwargs
        Will either return:
            None (kill the whole line)
        OR
            the new content    
    """
    if content == None:
        return None



def identity_filter(content):
    return content



FILTER_DICT = {'identity': identity_filter}



# ============================================================
# =                        LOGIC BLOCK                       =
# ============================================================

def process_jsonl(jsonl_path, config_dict, output_dir):
    """ Processes a full jsonl file and writes the output processed jsonl file
        (or if the filtering kills all the lines, will output nothing)
    """
    # Read file 
    lines = [json.loads(_) for _ in gzip.decompress(open(jsonl_path, 'rb').read()).splitlines()]

    # Process all lines
    output_lines = []
    for line in lines:
        output_content = process_content(line['content'], config)
        if output_content != None:
            line['content'] = output_content
            output_lines.append(line)
        else:
            continue

    # Save to output
    if len(output_lines) == 0:
        return
    output_file = os.path.join(output_dir, os.path.basename(jsonl_path))
    with open(output_file, 'wb') as f:
        f.write(gzip.compress(b'\n'.join([json.dumps(_).encode('utf-8') for _ in output_lines])))



def process_content(content, config):
    for filter_dict in config['pipeline']:
        filter_fxn = FILTER_DICT[filter_dict['fxn']]
        kwargs = {k: v for k,v in filter_fxn.items() if k != 'fxn'}
        content = filter_fxn(content, **kwargs)
        if content == None:
            return None
    return content


# =============================================================
# =                        MAIN BLOCK                         =
# =============================================================


def main(config_path, input_dir, output_dir, num_cpus=None):
    start_time = time.time()
    if num_cpus == None:
        num_cpus = os.cpu_count()

    files = glob.glob(os.path.join(input_dir, '**/*.jsonl.gz'), recursive=True)
    os.path.makedir(output_dir, exist_ok=True)
    config_dict = parse_config(config_path)

    partial_fxn = partial(process_jsonl, config_dict=config_dict, output_dir=output_dir)
    run_imap_multiprocessing(partial_fxn, files, num_cpus)

    print("Processed %s files in %.02f seconds" % (len(files), time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    # Add arguments
    parser.add_argument('--config', type=str, required=True,
                      help='location of the config.yaml')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='location of the input jsonl.gz files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='location of where the output jsonl.gz files will go')    
    parser.add_argument('--num-cpus', type=int,  required=False,
                        help='How many cpus to process using. Defaults to number of cpus on this machine')    
    args = parser.parse_args()

    main(args.config, args.input_dir, args.output_dir, num_cpus=args.num_cpus)


