import gradio as gr
import torchaudio
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from whisper.normalizers import EnglishTextNormalizer

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-tiny.en", task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path="/home/ubuntu/open_whisper/checkpoints/huggingface"
)
normalizer = EnglishTextNormalizer()
model.eval()


def transcribe(audio_file):
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_file)

    # Resample to the sample rate required by the model (if necessary)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        waveform = resampler(waveform)

    input_values = processor(
        waveform.squeeze(0), return_tensors="pt", sampling_rate=16000
    )["input_features"]

    with torch.no_grad():
        predicted_tokens = model.generate(input_values)

    transcription = processor.batch_decode(predicted_tokens)[0]
    normalized_transcription = normalizer(transcription)

    return normalized_transcription


demo = gr.Blocks()

mic_transcribe = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs=gr.Textbox(),
)

file_transcribe = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs=gr.Textbox(),
)

with demo:
    gr.TabbedInterface(
        [mic_transcribe, file_transcribe],
        ["Transcribe Microphone", "Transcribe Audio File"],
    )

demo.launch(share=True)
