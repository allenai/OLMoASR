# using whisper's decoding function
import whisper

model = whisper.load_model("checkpoints/tiny-en.pt")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio(
    "data/sanity-check/audio/eh77AUKedyM/segments/00:00:01.501_00:00:30.071.m4a"
)
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

options = whisper.DecodingOptions(language="en", without_timestamps=True)

result = whisper.decode(model, mel, options)

print(result.text)

