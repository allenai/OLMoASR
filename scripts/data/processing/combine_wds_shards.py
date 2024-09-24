import os
from itertools import islice
import multiprocessing
from tqdm import tqdm
import glob


# Function to move every 30 files into a new directory
def move_files_to_directory(tar_chunk, output_dir, batch_index):
    batch_dir = os.path.join(output_dir, f"{batch_index:05}")
    print(f"Creating directory {batch_dir}")
    os.makedirs(batch_dir, exist_ok=True)

    for file in tar_chunk:
        dest_file = os.path.join(batch_dir, os.path.basename(file))
        os.rename(file, dest_file)  # Move file to the new directory
        print(f"Moved {file} to {dest_file}")


# Function to group files in chunks of 30
def chunked_iterable(iterable, chunk_size):
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk


# Wrapper function to handle multiprocessing arguments
def move_files_in_parallel(args):
    return move_files_to_directory(*args)


# Main function to process files in batches of 30 using multiprocessing
def move_tar_files(source_dir, output_dir, chunk_size=30):
    # Get the list of tar files
    tar_files = sorted(glob.glob(os.path.join(source_dir, "*.tar")))
    
    # Prepare arguments for each chunk of tar files
    tasks = [
        (tar_chunk, output_dir, index)
        for index, tar_chunk in enumerate(chunked_iterable(tar_files, chunk_size))
    ]

    # Use multiprocessing to parallelize the file moving
    with multiprocessing.Pool() as pool:
        res = list(
            tqdm(pool.imap_unordered(move_files_in_parallel, tasks), total=len(tasks))
        )


# Example usage
if __name__ == "__main__":
    source_directory = "/weka/huongn/ow_440K_wds/"
    output_directory = "/weka/huongn/ow_440K_wds/"

    # Move every 30 tar files into separate directories using multiprocessing
    move_tar_files(source_directory, output_directory, chunk_size=30)
