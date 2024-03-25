import torch
import multiprocessing
import numpy as np
import os
from open_whisper import audio
from tqdm import tqdm


DEVICE = torch.device("cuda:0")


def get_mel(audio_file):
    audio_arr = audio.load_audio(audio_file, sr=16000)
    audio_arr = audio.pad_or_trim(audio_arr)
    mel_spec = audio.log_mel_spectrogram(audio_arr, device=DEVICE)

    return mel_spec


if __name__ == "__main__":
    audio_files = []
    for root, dirs, files in os.walk("data/audio"):
        if "segments" in root:
            audio_files.extend([os.path.join(root, f) for f in os.listdir(root)])

    rng = np.random.default_rng(42)

    sample = rng.choice(audio_files, 5000, replace=False)

    multiprocessing.set_start_method("spawn")

    with multiprocessing.Pool() as pool:
        log_mels = list(tqdm(pool.imap_unordered(get_mel, sample), total=len(sample)))

    log_mels_tensor = torch.stack(log_mels)

    print(f"shape: {log_mels_tensor.shape}")
    print(f"mean: {torch.mean(log_mels_tensor)}")
    print(f"min: {torch.min(log_mels_tensor)}")
    print(f"max: {torch.max(log_mels_tensor)}")
