from typing import Optional
import numpy as np
from typing import Optional, List, Tuple
import soundfile as sf
import os
from datasets import load_dataset, Dataset
import multiprocessing
from tqdm import tqdm
from itertools import repeat
import re
from whisper.tokenizer import get_tokenizer
from fire import Fire
import glob
import json

shared_ds = None

def init_worker(obj):
    global shared_ds
    shared_ds = obj

def extract_id(utt_id):
    match = re.search(r"^(.*?)-\d{5}(?!\d)", utt_id)

    result = match.group(1)
    return result

def check_over_ctx_len(
    timestamps: List,
    transcript_list: List,
    language: Optional[str] = None,
    last_seg: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Check if transcript text exceeds model context length

    Check if the total number of tokens in the transcript text exceeds the model context length

    Args:
        timestamps: List of timestamps
        transcript: Transcript as a dictionary

    Returns:
        True if the transcript text exceeds the model context length, False otherwise
    """
    try:
        if language is None:
            tokenizer = get_tokenizer(multilingual=False)
        else:
            tokenizer = get_tokenizer(language=language, multilingual=True)

        text_tokens = [
            (tokenizer.encode(" " + text.strip()))
            for i, text in enumerate(transcript_list)
        ]

        num_timestamp_tokens = (
            (len(timestamps) * 2) + 1 if not last_seg else (len(timestamps) * 2)
        )  # next_start timestamp (+1) when not last_seg
        num_text_tokens = sum([len(token_group) for token_group in text_tokens])
        num_tokens_ts_mode = num_timestamp_tokens + num_text_tokens + 2  # sot + eot
        num_tokens_no_ts_mode = num_text_tokens + 3  # sot + notimestamps + eot

        if num_tokens_ts_mode > 448 and num_tokens_no_ts_mode > 448:
            return True, None
        elif num_tokens_ts_mode > 448 and num_tokens_no_ts_mode <= 448:
            return False, {
                "ts_mode": False,
                "no_ts_mode": True,
                "num_tokens_no_ts_mode": num_tokens_no_ts_mode,
                "num_tokens_ts_mode": num_tokens_ts_mode,
            }
        elif num_tokens_ts_mode <= 448 and num_tokens_no_ts_mode > 448:
            return False, {
                "ts_mode": True,
                "no_ts_mode": False,
                "num_tokens_no_ts_mode": num_tokens_no_ts_mode,
                "num_tokens_ts_mode": num_tokens_ts_mode,
            }
        else:
            return False, {
                "ts_mode": True,
                "no_ts_mode": True,
                "num_tokens_no_ts_mode": num_tokens_no_ts_mode,
                "num_tokens_ts_mode": num_tokens_ts_mode,
            }
    except RuntimeError:
        return True, "error"
    except Exception as e:
        return True, "error"
    
def reseg_data(ds):
    a = 0
    b = 0
    utt_id_list = sorted(zip(ds["utt_id"], ds["id"]), key=lambda x: x[0])
    segments = []
    seg_count = 0
    with tqdm(total=ds.num_rows) as pbar:
        while b < len(utt_id_list) + 1:
            dur = extract_ts(utt_id_list[b][0], "end") - extract_ts(utt_id_list[a][0], "start")
            if dur <= 30.00:
                if extract_id(utt_id_list[b][0]) == extract_id(utt_id_list[a][0]):
                    b += 1
                    # ls_30_dur = dur
                else:
                    seg_count += 1
                    max_30_segs = [tpl[-1] for tpl in utt_id_list[a:b]]
                    segments.append([max_30_segs, seg_count - 1])
                    a = b
                    seg_count = 0
            else:
                seg_count += 1
                max_30_segs = [tpl[-1] for tpl in utt_id_list[a:b + 1]]
                segments.append([max_30_segs, seg_count - 1])
                if a == b:
                    a += 1
                    b += 1
                else:
                    a = b
            
            if b == len(utt_id_list):
                seg_count += 1
                max_30_segs = [tpl[-1] for tpl in utt_id_list[a:b + 1]]
                segments.append([max_30_segs, "last_seg", seg_count - 1])
                break
            pbar.update(1)
    
    return segments
        
def extract_ts(utt_id: str, type: Optional[str] = None, global_start: Optional[float] = None):
    raw_start, raw_end = utt_id.split("-")[-2:]
    start, end = int(raw_start[:-2]) + float(f'0.{raw_start[-2:]}'), int(raw_end[:-2]) + float(f'0.{raw_end[-2:]}')
    if global_start is not None:
        start -= global_start
        end -= global_start
        
    if type == "start":
        return start
    elif type == "end":
        return end
    else:
        return (start, end)

def extract_ts_list(utt_id_list):
    global_start = extract_ts(utt_id_list[0], "start")
    return [extract_ts(utt_id, None, global_start) for utt_id in utt_id_list]
    
def gen_new_seg(new_seg_idxs, output_dir):
    try:
        comb_seg = shared_ds[new_seg_idxs[0]]
    except Exception as e:
        return None
    
    idx = new_seg_idxs[-1]
    last_seg = True if len(new_seg_idxs) == 3 else False
    utt_id_list = comb_seg["utt_id"]
    new_utt_id = f"{'-'.join(utt_id_list[0].split('-')[:-3])}-{idx:05}-{utt_id_list[0].split('-')[-2]}-{utt_id_list[-1].split('-')[-1]}"
    new_audio_path = f"{output_dir}/{'-'.join(utt_id_list[0].split('-')[:-3])}-{idx:05}-{utt_id_list[0].split('-')[-2]}-{utt_id_list[-1].split('-')[-1]}.wav"
    
    audio_dicts = comb_seg["audio"]
    concatenated_audio = np.concatenate([d["array"] for d in audio_dicts])
    
    if concatenated_audio.shape[0] > 480000:
        concatenated_audio = concatenated_audio[:480000]
        
    sf.write(new_audio_path, concatenated_audio, 16000)

    text_list = comb_seg["text"][:-1] if len(comb_seg["text"]) > 1 else comb_seg["text"]
    ts_list = extract_ts_list(utt_id_list[:-1] if len(utt_id_list) > 1 else utt_id_list)
    over_ctx_len, res = check_over_ctx_len(ts_list, text_list, None, last_seg)
    # return ts_list, text_list, last_seg
    
    if over_ctx_len is True:
        return None
    else:
        return {
            "utt_id": new_utt_id,
            "audio": new_audio_path,
            "text": text_list,
            "ts": ts_list,
            "dur": concatenated_audio.shape[0] / 16000,
            "ts_mode": res["ts_mode"],
            "no_ts_mode": res["no_ts_mode"],
        }
        
def parallel_gen_new_seg(args):
    return gen_new_seg(*args)

def main(input_dir, audio_output_dir, text_output_dir):
    os.makedirs(audio_output_dir, exist_ok=True)
    os.makedirs(text_output_dir, exist_ok=True)
    
    # ds = load_dataset(path=input_dir, split="train")
    arrow_files = glob.glob(f"{input_dir}/*/*/*.arrow")
    print(f"{len(arrow_files)=}")
    print(f"{arrow_files[:5]=}")
    for i, arrow_file in enumerate(arrow_files):
        if not os.path.exists(f"{text_output_dir}/shard_seg_{i:05}.jsonl"):
            ds = Dataset.from_file(arrow_file)
            print(ds)
            
            print(f"Resegmenting data")
            segments = reseg_data(ds)
            
            shared_ds = ds
            
            print(f"Generating new segments")
            with multiprocessing.Pool(initializer=init_worker, initargs=(shared_ds,)) as pool:
                new_segments = list(tqdm(pool.imap_unordered(parallel_gen_new_seg, zip(segments, repeat(audio_output_dir))), total=len(segments)))
                
            new_segments = [seg for seg in new_segments if seg is not None]
            
            print("Writing new segments to disk")
            with open(f"{text_output_dir}/shard_seg_{i:05}.jsonl", "w") as f:
                for item in new_segments:
                    f.write(json.dumps(item) + "\n")
        else:
            print(f"shard_seg_{i:05}.jsonl already exists, skipping")
            continue
    print("Done resegmenting data")
    
if __name__ == "__main__":
    Fire(main)