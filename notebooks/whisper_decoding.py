# %%
import torch
import torch.nn.functional as F
from open_whisper import (
    load_audio,
    log_mel_spectrogram,
    pad_or_trim,
    load_model, 
    decoding,
    transcribe
)
from whisper import whisper

# %%
device = torch.device("cuda")

# %%
fp = "checkpoints/archive/sunny-tree-79/tiny-en-non-ddp_tiny-en_ddp-train_grad-acc_fp16_subset=full_lr=0.0015_batch_size=8_workers=18_epochs=25_train_val_split=0.99_inf.pt"
model = load_model(fp, device=device, inference=True)


# %%
model.to(device)

# %%
audio = load_audio(
    "data/eval/test-clean-librispeech/test-clean/1089/134686/1089-134686-0000.flac"
)
audio = pad_or_trim(audio)
audio_input = log_mel_spectrogram(audio).to(device)

# %%
options = decoding.DecodingOptions(language="en", without_timestamps=True)

# %%
result = decoding.decode(model, audio_input, options)
result.text

#%%
# below is code used to figure out that the bug was due to misalignment in choice of token 
# and vocab size covered by model. using the gpt-tiktoken tokenizer to decode output from
# model that was trained with the multilingal version of tokenizer (preprocessing) will result
# in incorrect output

# %%
batch_audio_input = audio_input.view(1, audio_input.shape[0], -1)

# %%
temp_tokenizer = whisper.tokenizer.get_tokenizer(
    multilingual=True, language="en", task="transcribe"
)

# %%
temp_tokenizer.sot_sequence_including_notimestamps

# %%
temp_tokenizer.eot

# %%
decoder_input = torch.full((1, 1), 50258, dtype=torch.long, device=device)

# %%
logits = model(batch_audio_input, decoder_input)
probs = F.softmax(logits, dim=-1)
# not a 1-dim tensor! grows as decoding continues
next_token_pred = torch.argmax(probs, dim=-1)
next_token_pred

# %%
decoder_input = torch.tensor([[50258, 50259, 50359, 50363]], device=device)
decoder_input

# %%
logits = model(batch_audio_input, decoder_input)
probs = F.softmax(logits, dim=-1)
# not a 1-dim tensor! grows as decoding continues
next_token_pred = torch.argmax(probs, dim=-1)
next_token_pred

# %%
probs.shape

# %%
decoder_input = torch.tensor([[50258, 50259]], device=device)
decoder_input

# %%
logits = model(batch_audio_input, decoder_input)
probs = F.softmax(logits, dim=-1)
# not a 1-dim tensor! grows as decoding continues
next_token_pred = torch.argmax(probs, dim=-1)
next_token_pred
