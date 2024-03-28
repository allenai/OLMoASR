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
)
from whisper import whisper

# %%
device = torch.device("cuda")
n_text_ctx = 448

# %%
fp = "checkpoints/archive/comic-cloud-73/tiny-en-non-ddp_tiny-en_ddp-train_grad-acc_fp16_subset=full_lr=0.0015_batch_size=8_workers=18_epochs=25_train_val_split=0.99.pt"
checkpoint = torch.load(fp, map_location=device)

# %%
dims = ModelDimensions(**checkpoint["dims"])
model = Whisper(dims)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# %%
audio = load_audio(
    "/home/ubuntu/open_whisper/data/eval/artie-bias-corpus/common_voice_en_12192.mp3"
)
audio = pad_or_trim(audio)
audio_input = log_mel_spectrogram(audio).to(device)
audio_input = audio_input.view(1, audio_input.shape[0], -1)
assert audio_input.shape == (1, 80, 3000)

# %%
model_tokenizer = tokenizer.get_tokenizer(
    multilingual=True, language="en", task="transcribe"
)
decoder_input = torch.full((1, 1), model_tokenizer.sot, dtype=torch.long, device=device)

# %%
generated_sequences = [[]]
active = torch.ones(1, dtype=torch.bool)

# %%
# decoding
while active.any():
    with torch.no_grad():
        logits = model(audio_input, decoder_input[:, : n_text_ctx - 1])
        probs = F.softmax(logits, dim=-1)
        # not a 1-dim tensor! grows as decoding continues
        next_token_pred = torch.argmax(probs, dim=-1)
        for i in range(1):
            if active[i] and len(generated_sequences[i]) < n_text_ctx - 1:
                generated_sequences[i].append(next_token_pred[i][-1].item())
                if next_token_pred[i][-1].item() == model_tokenizer.eot:
                    active[i] = False
            elif active[i] and len(generated_sequences[i]) == n_text_ctx - 1:
                active[i] = False
        if not active.any():
            break
        decoder_input = torch.cat(
            [
                decoder_input,
                next_token_pred[:, -1].unsqueeze(1),
            ],
            dim=-1,
        )
        print(model_tokenizer.decode(decoder_input[0]))

# %%
model_tokenizer.decode(decoder_input[0])
# %%
