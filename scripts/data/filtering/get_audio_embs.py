from open_whisper import load_model
import torch
import numpy as np
from whisper import audio
import glob
from torch.utils.data import Dataset, DataLoader
from typing import List
from tqdm import tqdm

ckpt = "/weka/huongn/ow_ckpts/base_15e4_440K_bs32_ebs768_12workers_5pass_FULL_SHARD_010325_vwngnkcy/eval_latesttrain_00349526_base_fsdp-train_grad-acc_bfloat16_inf.pt"
device = torch.device("cuda:0")
model = load_model(name=ckpt, device=device, inference=True, in_memory=True)
model.eval()
audio_segs = glob.glob("/weka/huongn/ow_seg/00013837/*/*.npy")
#%%
class AudioDataset(Dataset):
    """Dataset for audio and transcript segments

    Attributes:
        audio_files: List of audio file paths
        transcript_files: List of transcript file paths
        n_text_ctx: Number of text tokens
    """

    def __init__(
        self,
        audio_files: List[str],
    ):
        self.audio_files = audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(
        self, index
    ):
        audio_input = self.preprocess_audio(self.audio_files[index])

        return audio_input

    def preprocess_audio(self, audio_file: str):
        """Preprocesses the audio data for the model.

        Loads the audio file, pads or trims the audio data, and computes the log mel spectrogram.

        Args:
            audio_file: The path to the audio file

        Returns:
            A tuple containing the name of audio file and the log mel spectrogram
        """
        audio_arr = np.load(audio_file).astype(np.float32) / 32768.0
        audio_arr = audio.pad_or_trim(audio_arr)
        mel_spec = audio.log_mel_spectrogram(audio_arr)

        return mel_spec

audio_dataset = AudioDataset(audio_files=audio_segs)
# %%
dataloader = DataLoader(
    audio_dataset,
    batch_size=32,
    pin_memory=True,
    num_workers=8,
    drop_last=False,
    shuffle=False,
    persistent_workers=True,
)
# %%
audio_embds = []
with torch.no_grad():
    for idx, batch in enumerate(tqdm(dataloader)):
        audio_input = batch
        audio_input = audio_input.to(device)
        audio_embds_batch = model.embed_audio(audio_input)
        audio_embds.append(audio_embds_batch)
torch.save(torch.cat(audio_embds, dim=0), "/weka/huongn/audio_embs.pt")
# %%
