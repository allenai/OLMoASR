from open_whisper import audio, tokenizer, preprocess, model
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from dataclasses import dataclass
import numpy as np
from torch.optim import AdamW

AUDIO_FILE = "data/audio/eh77AUKedyM/segments/00:00:01.501_00:00:30.071.wav"
TRANSCRIPT_FILE = "data/transcripts/eh77AUKedyM/segments/00:00:01.501_00:00:30.071.txt"
DEVICE = torch.device("cuda:0")

# encoder architecture
n_mels = 80
n_audio_ctx = 1500
n_audio_state = 384
n_audio_head = 6
n_audio_layer = 4

# decoder architecture
n_vocab = 51864
n_text_ctx = 448
n_text_state = 384
n_text_head = 6
n_text_layer = 4

# # Load the audio waveform
# audio_arr = audio.load_audio(AUDIO_FILE, sr=16000)
# # Pad or trim the audio array to N_SAMPLES, as expected by the encoder
# audio_arr = audio.pad_or_trim(audio_arr)
# # Convert to mel spectrogram
# # this results in a tensor of shape (80, 3000), but n_audio_ctx = 1500. maybe this is due to the conv1d layer (with stride 2 applied to spectrogram?)
# mel_spec = audio.log_mel_spectrogram(audio_arr, device=DEVICE)

# # not sure if this is the right way to normalize feature
# mel_spec_normalized = (mel_spec - mel_spec.mean()) / mel_spec.std()
# mel_spec_scaled = mel_spec_normalized / (mel_spec_normalized.abs().max())

# # Load transcript file
# with open(file=TRANSCRIPT_FILE, mode="r") as f:
#     transcript = f.read().strip()
# # Load the tokenizer
# # question - why when I set multilingual to False, the language and task tokens are set to None?
# # what is the @cached_property decorator?
# # how to have sot_sequence specify no decoding with timestamps
# tokenizer = tokenizer.get_tokenizer(multilingual=True, language="en", task="transcribe")
# # tokenize and encode text
# text_tokens = tokenizer.encode(transcript)
# # add start sequence and end tokens
# # sot/eot token only used when at first/last audio/transcript segment
# text_tokens = list(tokenizer.sot_sequence_including_notimestamps) + text_tokens
# # padding of text tokens
# text_tokens = np.pad(
#     text_tokens,
#     pad_width=(0, n_text_ctx - len(text_tokens)),
#     mode="constant",
#     constant_values=tokenizer.no_speech,
# )
# # convert text tokens to tensor
# text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=DEVICE)


class AudioDataset(Dataset):
    def __init__(self, audio_dir):
        self.audio_files = [
            os.path.join(audio_dir, audio_file) for audio_file in os.listdir(audio_dir)
        ]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        return self.preprocess_audio(self.audio_files[index])

    def preprocess_audio(self, audio_file):
        audio_arr = audio.load_audio(audio_file, sr=16000)
        audio_arr = audio.pad_or_trim(audio_arr)
        mel_spec = audio.log_mel_spectrogram(audio_arr, device=DEVICE)
        mel_spec_normalized = (mel_spec - mel_spec.mean()) / mel_spec.std()
        mel_spec_scaled = mel_spec_normalized / (mel_spec_normalized.abs().max())
        return mel_spec_scaled


class TextDataset(Dataset):
    def __init__(self, transcript_dir, tokenizer, n_text_ctx):
        self.transcript_files = [
            os.path.join(transcript_dir, transcript_file)
            for transcript_file in os.listdir(transcript_dir)
        ]
        self.tokenizer = tokenizer
        self.n_text_ctx = n_text_ctx

    def __len__(self):
        return len(self.transcript_files)

    def __getitem__(self, index):
        return self.preprocess_text(self.transcript_files[index], index)

    def preprocess_text(self, transcript_file, file_index):
        with open(file=transcript_file, mode="r") as f:
            transcript = f.read().strip()
        text_tokens = self.tokenizer.encode(transcript)

        if file_index == 0:
            text_tokens = (
                list(self.tokenizer.sot_sequence_including_notimestamps) + text_tokens
            )

        text_tokens = np.pad(
            text_tokens,
            pad_width=(0, self.n_text_ctx - len(text_tokens)),
            mode="constant",
            constant_values=self.tokenizer.no_speech,
        )

        if file_index == len(self.transcript_files) - 1:
            text_tokens = (
                text_tokens[: -len(self.tokenizer.no_speech)] + self.tokenizer.eot
            )

        text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=DEVICE)
        return text_tokens


class AudioTextDataset(Dataset):
    def __init__(self, audio_dir, transcript_dir, tokenizer, device, n_text_ctx):
        self.audio_files = [
            os.path.join(audio_dir, audio_file) for audio_file in os.listdir(audio_dir)
        ]
        self.transcript_files = [
            os.path.join(transcript_dir, transcript_file)
            for transcript_file in os.listdir(transcript_dir)
        ]
        self.tokenizer = tokenizer
        self.device = device
        self.n_text_ctx = n_text_ctx

        if len(self.audio_files) != len(self.transcript_files):
            raise ValueError(
                "The number of audio files and transcript files must be the same"
            )

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_input = self.preprocess_audio(self.audio_files[index])
        text_tokens = self.preprocess_text(self.transcript_files[index], index)
        text_input = text_tokens[:-1]
        text_y = text_tokens[1:]
        return audio_input, text_input, text_y

    def preprocess_audio(self, audio_file):
        audio_arr = audio.load_audio(audio_file, sr=16000)
        audio_arr = audio.pad_or_trim(audio_arr)
        mel_spec = audio.log_mel_spectrogram(audio_arr, device=self.device)
        mel_spec_normalized = (mel_spec - mel_spec.mean()) / mel_spec.std()
        mel_spec_scaled = mel_spec_normalized / (mel_spec_normalized.abs().max())
        return mel_spec_scaled

    def preprocess_text(self, transcript_file, file_index):
        with open(file=transcript_file, mode="r") as f:
            transcript = f.read().strip()
        text_tokens = self.tokenizer.encode(transcript)

        if file_index == 0:
            text_tokens = (
                list(self.tokenizer.sot_sequence_including_notimestamps) + text_tokens
            )

        text_tokens = np.pad(
            text_tokens,
            pad_width=(0, self.n_text_ctx - len(text_tokens)),
            mode="constant",
            constant_values=self.tokenizer.no_speech,
        )

        if file_index == len(self.transcript_files) - 1:
            text_tokens = (
                text_tokens[: -len('<|nospeech|>')] + self.tokenizer.eot
            )

        text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=self.device)
        return text_tokens


tokenizer = tokenizer.get_tokenizer(multilingual=True, language="en", task="transcribe")
audio_text_dataset = AudioTextDataset(
    audio_dir="data/audio/eh77AUKedyM/segments",
    transcript_dir="data/transcripts/eh77AUKedyM/segments",
    tokenizer=tokenizer,
    device=DEVICE,
    n_text_ctx=448,
)
audio_text_dataloader = DataLoader(
    audio_text_dataset, batch_size=1, shuffle=True, num_workers=0
)


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


model_dims = ModelDimensions(
    n_mels=80,
    n_audio_ctx=1500,
    n_audio_state=384,
    n_audio_head=6,
    n_audio_layer=4,
    n_vocab=51864,
    n_text_ctx=448,
    n_text_state=384,
    n_text_head=6,
    n_text_layer=4,
)


model = model.Whisper(dims=model_dims).to(DEVICE)
optimizer = AdamW(model.parameters())
model.train()

for batch_idx, batch in enumerate(audio_text_dataloader):
    optimizer.zero_grad()
    audio_input, text_input, text_y = batch

    logits = model(audio_input, text_input)
    # probs = F.softmax(logits, dim=-1)
    # pred = torch.argmax(probs, dim=-1)

    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), text_y.view(-1))

    loss.backward()
    optimizer.step()
    


