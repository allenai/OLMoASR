from get_eval_train import get_eval_train
import os
import numpy as np
import glob
import json
from open_whisper.utils import TranscriptReader
from torchaudio.datasets import TEDLIUM
from fire import Fire

# preprocess both positive and negative training data to ensure they are in the same format (no transcript specific format remains)
# generate text file w/ labels from these 2 sets of data

def main(eval_set: str, eval_train_dir: str, train_dir: str, segment_filter: bool):
    # collect all positive training data (data from eval set)
    if eval_set == "tedlium":
        if not os.path.exists(f"{eval_train_dir}/TEDLIUM_release-3"):
            get_eval_train(eval_set=eval_set, eval_dir=eval_train_dir)
            
        # Initialize the dataset
        self.dataset = TEDLIUM(
            root=f"{eval_train_dir}", release="release3", subset="train"
        )

        # Specify the output text file
        output_file = f"{eval_train_dir}/tedlium_train.txt"

        # Open the file for writing
        with open(output_file, "w", encoding="utf-8") as file:
            for index in range(len(self.dataset)):
                # Get the data for the current index
                _, _, text_y, *_ = self.dataset[index]
                
                # Write the transcript to the file
                file.write(text_y + "\n")

        print(f"Transcripts have been written to {output_file}.")
        
        # get count of documents in eval train data
        negative_subsample_count = len(self.dataset)
        print(f"Number of documents in eval train data: {negative_subsample_count}")
    
    # subsample negative training data (from training pool) and match num. of docs w/ positive training data (from samples dict)
    # if segment_filter:
    #     samples_dicts_dirs = glob.glob(f"{train_dir}/*")
    #     sample_dicts_dir = np.random.choice(samples_dicts_dirs, 1)[0]
    #     with open(f"{samples_dicts_dir}/samples_dicts.jsonl", "r") as f:
    #         sample_dicts = [json.loads(line.strip()) for line in f]
    # else:
    #     shard_dirs = glob.glob(f"{train_dir}/*")
    #     subsampled_train_data = []
    #     subsampled_count = 0
    #     while True:
    #         shard_dir = np.random.choice(shard_dirs, 1)[0]
    #         ext = "vtt" if int(shard_dir.split("/")[-1]) > 2448 else "srt"
    #         transcript_files = glob.glob(f"{shard_dir}/*/*.{ext}")
    #         if len(transcript_files) < (negative_subsample_count - subsampled_count):
    #             subsampled_train_data.extend(transcript_files)
    #             subsampled_count += len(transcript_files)
    #         elif subsampled_count < negative_subsample_count:
    #             subsampled_train_data.extend(np.random.choice(transcript_files, negative_subsample_count - subsampled_count, replace=False))
    #             break
    #         else:
    #             subsampled_train_data.extend(np.random.choice(transcript_files, negative_subsample_count, replace=False))
    #             break
        
if __name__ == "__main__":
    Fire(main) 

    
    
    
        