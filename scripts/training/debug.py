from open_whisper import audio, tokenizer, preprocess
import torch
import torch.nn.functional as F

AUDIO_FILE = "data/audio/eh77AUKedyM/segments/00:00:01.501_00:00:30.071.wav"
TRANSCRIPT_FILE = "data/transcripts/eh77AUKedyM/segments/00:00:01.501_00:00:30.071.txt"
DEVICE = "cuda:0"

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

# Load the audio waveform
audio_arr = audio.load_audio(AUDIO_FILE, sr=16000)
# Pad or trim the audio array to N_SAMPLES, as expected by the encoder
audio_arr = audio.pad_or_trim(audio_arr)
# Convert to mel spectrogram
# this results in a tensor of shape (80, 3000), but n_audio_ctx = 1500. maybe this is due to the conv1d layer (with stride 2 applied to spectrogram?)
mel_spec = audio.log_mel_spectrogram(audio_arr, device=DEVICE)

# not sure if this is the right way to normalize feature
mel_spec_normalized = (mel_spec - mel_spec.mean()) / mel_spec.std()
mel_spec_scaled = mel_spec_normalized / (mel_spec_normalized.abs().max())

# Load transcript file
with open(file=TRANSCRIPT_FILE, mode="r") as f:
    transcript = f.read().strip()
# Load the tokenizer
# question - why when I set multilingual to False, the language and task tokens are set to None?
# what is the @cached_property decorator?
# how to have sot_sequence specify no decoding with timestamps
tokenizer = tokenizer.get_tokenizer(multilingual=True, language="en", task="transcribe")
# tokenize and encode text
text_tokens = tokenizer.encode(transcript)
text_tokens = (
    list(tokenizer.sot_sequence_including_notimestamps) + text_tokens + [tokenizer.eot]
)
text = tokenizer.decode(text_tokens)  # for debugging
# convert text tokens to tensor
text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=DEVICE)
# padding of text tokens
text_tokens = F.pad(
    text_tokens, pad=(0, n_text_ctx - len(text_tokens)), mode="constant", value=0
)


