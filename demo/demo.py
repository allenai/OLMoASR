import gradio as gr
import torchaudio
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from whisper.normalizers import EnglishTextNormalizer

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small.en", task="transcribe"
)
model = WhisperForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path="/home/ubuntu/open_whisper/checkpoints/best_small_hf_demo"
)
model.cuda()
normalizer = EnglishTextNormalizer()
model.eval()


def stereo_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    # Check if the waveform is stereo
    if waveform.shape[0] == 2:
        # Average the two channels to convert to mono
        mono_waveform = waveform.mean(dim=0, keepdim=True)
        return mono_waveform
    else:
        # If already mono, return as is
        return waveform


def transcribe(audio_file):
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_file)
    waveform = stereo_to_mono(waveform)
    print(waveform.shape)

    # Resample to the sample rate required by the model (if necessary)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        waveform = resampler(waveform)

    if waveform.shape[1] < 480000:  # shortform
        inputs = processor(
            waveform.squeeze(0), return_tensors="pt", sampling_rate=16_000
        )["input_features"]

        inputs = inputs.to("cuda", torch.float32)
    
        with torch.no_grad():
            predicted_tokens = model.generate(inputs)
         
        transcription = processor.batch_decode(
            predicted_tokens, skip_special_tokens=True
        )
    else: # longform
        inputs = processor(
            waveform.squeeze(0),
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
            sampling_rate=16_000,
        )

        inputs = inputs.to("cuda", torch.float32)

        with torch.no_grad():
            predicted_tokens = model.generate(**inputs, return_segments=True)

        transcription = processor.batch_decode(
            predicted_tokens["sequences"], skip_special_tokens=True
        )
        
    normalized_transcription = normalizer(transcription[0])

    return transcription[0], normalized_transcription


demo = gr.Blocks()

mic_transcribe = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs=[gr.Textbox(label="Transcription"), gr.Textbox(label="Normalized Transcription")],
)

file_transcribe = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs=[gr.Textbox(label="Transcription"), gr.Textbox(label="Normalized Transcription")],
)

with demo:
    gr.TabbedInterface(
        [mic_transcribe, file_transcribe],
        ["Transcribe Microphone", "Transcribe Audio File"],
    )

demo.launch(share=True)
