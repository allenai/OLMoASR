import gradio as gr
from gradio_rich_textbox import RichTextbox
import torchaudio
import re
import librosa
import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from whisper.normalizers import EnglishTextNormalizer
from bs4 import BeautifulSoup

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_path = "/home/ubuntu/open_whisper/checkpoints/medium_hf_demo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_path)

normalizer = EnglishTextNormalizer()
model.eval()

transcriber = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    chunk_length_s=30,
)


def stereo_to_mono(waveform):
    # Check if the waveform is stereo
    if waveform.shape[0] == 2:
        # Average the two channels to convert to mono
        mono_waveform = np.mean(waveform, axis=0)
        return mono_waveform
    else:
        # If already mono, return as is
        return waveform


def is_silent_rms(waveform, threshold=1e-4):
    rms = np.sqrt(np.mean(waveform**2))
    return rms < threshold


def transcribe(audio_file, timestamp_text, transcription_text):
    waveform, sample_rate = librosa.load(audio_file, sr=None, mono=False)
    waveform = stereo_to_mono(waveform)
    print(waveform.shape)

    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)

    result = transcriber(waveform, return_timestamps=True)
    print(f"{result['text']=}\n")
    print(f"{result['chunks']=}\n")

    text = result["text"].strip().replace("\n", " ")
    text = re.sub(r"(foreign|foreign you)\s*$", "", text)

    chunks = process_chunks(result["chunks"])

    # Edit components
    transSoup = BeautifulSoup(transcription_text, "html.parser")
    transText = transSoup.find(id="transcriptionText")
    if transText:
        transText.clear()
        transText.append(BeautifulSoup(text, "html.parser"))

    timeSoup = BeautifulSoup(timestamp_text, "html.parser")
    timeText = timeSoup.find(id="timestampText")
    if timeText:
        timeText.clear()
        timeText.append(BeautifulSoup(chunks, "html.parser"))

    return str(timeSoup), str(transSoup)


def process_chunks(chunks):
    processed_chunks = []
    for chunk in chunks:
        text = chunk["text"].strip()
        if not re.match(r"(foreign|foreign you)\s*$", text):
            if text.strip() == "":
                continue
            start = chunk["timestamp"][0]
            end = chunk["timestamp"][1]
            pattern = r"\n(?!\d+\.\d+\s*-->)"
            text = re.sub(pattern, " ", text)
            processed_chunks.append(f"{start:.2f} --> {end:.2f}: {text.strip()} <br>")
        else:
            break
    return "\n".join(processed_chunks)


event_process_js = """
<script>
function getTime() {
    lastIndex = -1;
    setInterval(() => {
        time = document.getElementById('time');
        timestampText = document.getElementById('timestampText');
        if(timestampText) {
            if(timestampText.innerText != '') {
                if(time == null) {
                    timestampText.innerText = '';
                    transcriptionText = document.getElementById('transcriptionText');
                    if(transcriptionText) {
                        transcriptionText.innerText = '';
                    }
                    lastIndex = -1;
                }
                if(time != null && timestampText != null) {
                    timeContent = time.textContent;
                    const parts = timeContent.split(":").map(Number);
                    currTime = parseFloat(parts[0]) * 60 + parseFloat(parts[1]);            
                    currText = timestampText.innerText;
                    const matches = [...currText.matchAll(/([\d.]+)\s*-->/g)];
                    const startTimestamps = matches.map(m => parseFloat(m[1]));
                    
                    if(startTimestamps.length != 0) {
                        correctIndex = 0;
                        for (let i = 0; i < startTimestamps.length; i++) {
                            if (startTimestamps[i] <= currTime) {
                                correctIndex = i;
                            }
                            else {
                                break;
                            }
                        }
                        if (lastIndex != correctIndex) {
                            lastIndex = correctIndex;
                            lines = currText.split('\\n');
                            lines[correctIndex] = '<span style="background-color: #ff69b4; padding: 3px 8px; font-weight: 500; border-radius: 4px; color: white; box-shadow: 0 0 10px rgba(255, 105, 180, 0.5);">' + lines[correctIndex] + '</span>';
                            try {
                                timestampText.innerHTML = lines.join('<br>');
                            }
                            catch (e) {
                                console.log('Not Updating!');
                            }
                        }
                    }
                }
                
            }
        }
    }, 50);
}
setTimeout(getTime, 1000);
</script>
"""

demo = gr.Blocks(head=event_process_js, theme=gr.themes.Default(primary_hue="emerald", secondary_hue="green"))
with demo:
    audio = gr.Audio(sources=["upload", "microphone"], type="filepath")
    button = gr.Button(
        "Transcribe",
        variant="primary",
    )
    with gr.Row():
        timestampText = gr.HTML(
            """
            <div style="background: white; border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; width: 100%; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); flex: 1; margin-right: 10px;">
                <div style="color: #374151; font-size: 14px; font-weight: 500; margin-bottom: 8px;">Timestamp Text</div>
                <div id="timestampText"; style="color: #6b7280; font-size: 14px; line-height: 1.5; min-height: 100px; font-family: system-ui, sans-serif;"></div>
            </div>
            """
        )

        transcriptionText = gr.HTML(
            """
            <div style="background: white; border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; width: 100%; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); flex: 1; margin-left: 10px;">
                <div style="color: #374151; font-size: 14px; font-weight: 500; margin-bottom: 8px;">Transcription Text</div>
                <div id="transcriptionText"; style="color: #6b7280; font-size: 14px; line-height: 1.5; min-height: 100px; font-family: system-ui, sans-serif;"></div>
            </div>
            """
        )
    button.click(
        fn=transcribe,
        inputs=[audio, timestampText, transcriptionText],
        outputs=[timestampText, transcriptionText],
    )
demo.launch(share=True)
