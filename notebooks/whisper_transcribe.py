# %%
import torch
import torch.nn.functional as F
from olmoasr import (
    load_audio,
    log_mel_spectrogram,
    pad_or_trim,
    load_model,
    decoding,
    transcribe,
)
import whisper

# %%
device = torch.device("cuda")

# %%
fp = "checkpoints/whisper/tiny-en-whisper.pt"
model = load_model(fp, device=device, inference=True)

# %%
model.to(device)

# %%
transcribe.transcribe(
    model,
    "data/eval/test-clean-librispeech/test-clean/1089/134686/1089-134686-0000.flac",
)

# %%
fp = "checkpoints/archive/sunny-tree-79/tiny-en-non-ddp_tiny-en_ddp-train_grad-acc_fp16_subset=full_lr=0.0015_batch_size=8_workers=18_epochs=25_train_val_split=0.99_inf.pt"
model = load_model(fp, device=device, inference=True)

# %%
model.to(device)

# %%
transcribe.transcribe(
    model,
    "data/eval/test-clean-librispeech/test-clean/1089/134686/1089-134686-0000.flac",
)
