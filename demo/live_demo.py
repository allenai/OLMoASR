import gradio as gr
import torchaudio
import re
import librosa
import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from whisper.normalizers import EnglishTextNormalizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_path = "/home/ubuntu/open_whisper/checkpoints/medium_hf_demo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_path)

transcriber = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
    torch_dtype=torch_dtype,
)


def transcribe(stream, new_chunk):
    sr, y = new_chunk

    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
    text = transcriber({"sampling_rate": sr, "raw": stream})["text"]
    text = re.sub(r"(foreign|foreign you)\s*$", "", text)
    return stream, text


demo = gr.Blocks(
    theme=gr.themes.Default(primary_hue="emerald", secondary_hue="green"),
)
with demo:
    audio_source = gr.Audio(sources=["microphone"], streaming=True)
    text_output = gr.Textbox(label="Transcription")
    clear_btn = gr.Button("Clear")
    state = gr.State()
    audio_source.stream(
        fn=transcribe,
        inputs=[state, audio_source],
        outputs=[state, text_output],
    )
    clear_btn.click(
        fn=lambda: "", inputs=[], outputs=text_output  # returns empty string
    )
demo.launch(share=True)

# demo.launch(share=True)
