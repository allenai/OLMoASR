#%%
import os
from open_whisper.utils import TranscriptReader
import multiprocessing
from tqdm import tqdm
from whisper import audio, DecodingOptions, load_model
from whisper.normalizers import EnglishTextNormalizer
import jiwer
from open_whisper.config.model_dims import VARIANT_TO_DIMS
from torch.utils.data import Dataset, DataLoader
import torch

#%%
all_transcripts = []
for root, *_ in os.walk("data/transcripts"):
    if "segments" in root:
        all_transcripts.extend((os.path.join(root, path) for path in os.listdir(root)))

all_transcripts = sorted(all_transcripts)
subset_transcripts = all_transcripts[500000:]
# %%
print(len(all_transcripts))
#%%
all_audio = []
for root, *_ in os.walk("data/audio"):
    if "segments" in root:
        all_audio.extend((os.path.join(root, path) for path in os.listdir(root))) 

all_audio = sorted(all_audio)
subset_audio = all_audio[500000:]
# %%
print(len(all_audio))
# %%
device = torch.device("cuda")
model = load_model("medium.en", device=device)
normalizer = EnglishTextNormalizer()

# %%
class AudioTextDataset(Dataset):
    def __init__(self, audio_files, transcript_files):
        self.audio_files = audio_files
        self.transcript_files = transcript_files

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, index):
        audio_file, audio_input = self.preprocess_audio(self.audio_files[index])
        reader = TranscriptReader(self.transcript_files[index])
        t_dict, *_ = reader.read()
        text_y = reader.extract_text(t_dict)

        return (
            audio_file,
            audio_input,
            text_y,
        )
    
    def preprocess_audio(self, audio_file):
        audio_arr = audio.load_audio(audio_file, sr=16000)
        audio_arr = audio.pad_or_trim(audio_arr)
        mel_spec = audio.log_mel_spectrogram(audio_arr)
        return audio_file, mel_spec

#%%
dataset = AudioTextDataset(audio_files=subset_audio, transcript_files=subset_transcripts)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True, drop_last=False, persistent_workers=True)

# %%
options = DecodingOptions(language="en", without_timestamps=True)
for batch in tqdm(dataloader):
    with torch.no_grad():
        audio_files, audio_input, text_y = batch
        audio_input = audio_input.to(device)
        results = model.decode(audio_input, options=options)
        with open("logs/data/filtering/unrelated_text_5.txt", "a") as f:
            for i in range(len(results)):
                f.write(f"{audio_files[i]}||{normalizer(results[i].text)}||{normalizer(text_y[i])}\n")

# %%
