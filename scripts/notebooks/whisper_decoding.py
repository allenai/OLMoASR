# %%
import torch
import torch.nn.functional as F
from open_whisper import (
    ModelDimensions,
    load_audio,
    log_mel_spectrogram,
    pad_or_trim,
    Whisper,
    tokenizer,
    load_model
)
from whisper import whisper

# %%
fp = "/home/ubuntu/open_whisper/checkpoints/archive/comic-cloud-73/tiny-en-non-ddp_tiny-en_ddp-train_grad-acc_fp16_subset=full_lr=0.0015_batch_size=8_workers=18_epochs=25_train_val_split=0.99.pt"
model = whisper.load_model(fp)

# %%
device = torch.device("cuda")

# %%
model.to(device)

# %%
audio = load_audio(
    "/home/ubuntu/open_whisper/data/eval/artie-bias-corpus/common_voice_en_250.mp3"
)
audio = pad_or_trim(audio)
audio_input = log_mel_spectrogram(audio).to(device)

# %%
options = whisper.DecodingOptions(language="en", without_timestamps=True)

# %%
result = whisper.decode(model, audio_input, options)
result.text

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
