import gradio as gr
from gradio_rich_textbox import RichTextbox
import torchaudio
import re
import librosa
import torch
import numpy as np
import os
import tempfile
import subprocess
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from whisper.normalizers import EnglishTextNormalizer
from whisper import audio, DecodingOptions
from whisper.tokenizer import get_tokenizer
from whisper.decoding import detect_language
from olmoasr import load_model
from bs4 import BeautifulSoup

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Configuration for model download and conversion
OLMOASR_REPO = "olmoasr/OLMoASR-small.en"  # Temporary model link as requested
CHECKPOINT_FILENAME = (
    "latesttrain_00524288_small_ddp-train_grad-acc_fp16_non_ddp_inf.pt"  # Adjust based on actual filename in the repo
)
LOCAL_CHECKPOINT_DIR = "checkpoints"
HF_MODEL_DIR = "checkpoints/small_hf_converted"


def ensure_checkpoint_dir():
    """Ensure the checkpoint directory exists."""
    Path(LOCAL_CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(HF_MODEL_DIR).mkdir(parents=True, exist_ok=True)


def download_olmoasr_checkpoint():
    """Download OLMoASR checkpoint from HuggingFace hub."""
    ensure_checkpoint_dir()

    local_checkpoint_path = os.path.join(LOCAL_CHECKPOINT_DIR, CHECKPOINT_FILENAME)

    # Check if checkpoint already exists
    if os.path.exists(local_checkpoint_path):
        print(f"Checkpoint already exists at {local_checkpoint_path}")
        return local_checkpoint_path

    try:
        print(f"Downloading checkpoint from {OLMOASR_REPO}")
        downloaded_path = hf_hub_download(
            repo_id=OLMOASR_REPO,
            filename=CHECKPOINT_FILENAME,
            local_dir=LOCAL_CHECKPOINT_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded checkpoint to {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"Error downloading checkpoint: {e}")


def convert_checkpoint_to_hf(checkpoint_path):
    """Convert OLMoASR checkpoint to HuggingFace format using subprocess."""
    if os.path.exists(os.path.join(HF_MODEL_DIR, "config.json")):
        print(f"HuggingFace model already exists at {HF_MODEL_DIR}")
        return HF_MODEL_DIR

    try:
        print(f"Converting checkpoint {checkpoint_path} to HuggingFace format")

        # Path to the conversion script
        script_path = os.path.join(os.path.dirname(__file__), "convert_openai_to_hf.py")

        # Run the conversion script using subprocess
        cmd = [
            sys.executable,
            script_path,
            "--checkpoint_path",
            checkpoint_path,
            "--pytorch_dump_folder_path",
            HF_MODEL_DIR,
            "--convert_preprocessor",
            "True",
        ]

        print(f"Running conversion command: {' '.join(cmd)}")

        # Execute the conversion script
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        print("Conversion output:")
        print(result.stdout)

        if result.stderr:
            print("Conversion warnings/errors:")
            print(result.stderr)

        # Verify that the conversion was successful
        if os.path.exists(os.path.join(HF_MODEL_DIR, "config.json")):
            print(f"Model successfully converted and saved to {HF_MODEL_DIR}")
            return HF_MODEL_DIR
        else:
            raise Exception("Conversion completed but config.json not found")

    except subprocess.CalledProcessError as e:
        print(f"Conversion script failed with return code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise e
    except Exception as e:
        print(f"Error converting checkpoint: {e}")
        raise e


def initialize_models():
    """Initialize both HuggingFace and OLMoASR models."""
    # Download and convert HuggingFace model
    checkpoint_path = download_olmoasr_checkpoint()
    hf_model_path = convert_checkpoint_to_hf(checkpoint_path)
    olmoasr_ckpt = checkpoint_path

    # Load HuggingFace model
    hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        hf_model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    hf_model.to(device).eval()
    processor = AutoProcessor.from_pretrained(hf_model_path)

    # Load OLMoASR model
    olmoasr_model = load_model(
        name=olmoasr_ckpt, device=device, inference=True, in_memory=True
    )
    olmoasr_model.to(device).eval()

    return hf_model, processor, olmoasr_model


# Initialize models
print("Initializing models...")
hf_model, processor, olmoasr_model = initialize_models()
print("Models initialized successfully!")

normalizer = EnglishTextNormalizer()


def stereo_to_mono(waveform):
    # Check if the waveform is stereo
    if waveform.shape[0] == 2:
        # Average the two channels to convert to mono
        mono_waveform = np.mean(waveform, axis=0)
        return mono_waveform
    else:
        # If already mono, return as is
        return waveform


def hf_chunk_transcribe(audio_file, timestamp_text, transcription_text):
    hf_transcriber = pipeline(
        "automatic-speech-recognition",
        model=hf_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30,
    )

    waveform, sample_rate = librosa.load(audio_file, sr=None, mono=False)
    waveform = stereo_to_mono(waveform)
    print(waveform.shape)

    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)

    result = hf_transcriber(waveform, return_timestamps=True)
    print(f"{result['text']=}\n")
    print(f"{result['chunks']=}\n")

    # text = result["text"].strip().replace("\n", " ")
    # text = re.sub(r"(foreign|foreign you|you)\s*$", "", text)

    chunks, text = hf_process_chunks(result["chunks"])
    print(f"{chunks=}\n")
    print(f"{text=}\n")

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


def olmoasr_seq_transcribe(audio_file, timestamp_text, transcription_text):
    waveform, sample_rate = librosa.load(audio_file, sr=None, mono=False)
    waveform = stereo_to_mono(waveform)
    print(waveform.shape)

    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)

    options = dict(
        task="transcribe",
        language="en",
        without_timestamps=False,
        beam_size=5,
        best_of=5,
    )
    result = olmoasr_model.transcribe(waveform, verbose=False, **options)
    print(f"{result['text']=}\n")
    print(f"{result['segments']=}\n")

    # text = result["text"].strip().replace("\n", " ")
    # text = re.sub(r"(foreign|foreign you|Thank you for watching!|. you)\s*$", "", text)

    chunks, text = olmoasr_process_chunks(result["segments"])
    print(f"{chunks=}\n")
    print(f"{text=}\n")

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


def hf_seq_transcribe(audio_file, timestamp_text, transcription_text):
    hf_transcriber = pipeline(
        "automatic-speech-recognition",
        model=hf_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    waveform, sample_rate = librosa.load(audio_file, sr=None, mono=False)
    waveform = stereo_to_mono(waveform)
    print(waveform.shape)

    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)

    result = hf_transcriber(
        waveform,
        return_timestamps=True,
    )
    print(f"{result['text']=}\n")
    print(f"{result['chunks']=}\n")

    # text = result["text"].strip().replace("\n", " ")
    # text = re.sub(r"(foreign|foreign you|you)\s*$", "", text)

    chunks, text = hf_seq_process_chunks(result["chunks"])
    print(f"{text=}\n")
    print(f"{chunks=}\n")

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


def main_transcribe(inference_strategy, audio_file, timestamp_text, transcription_text):
    if inference_strategy == "HuggingFace Chunking":
        return hf_chunk_transcribe(audio_file, timestamp_text, transcription_text)
    elif inference_strategy == "OLMoASR Sequential":
        return olmoasr_seq_transcribe(audio_file, timestamp_text, transcription_text)
    elif inference_strategy == "HuggingFace Sequential":
        return hf_seq_transcribe(audio_file, timestamp_text, transcription_text)


def olmoasr_process_chunks(chunks):
    processed_chunks = []
    processed_chunks_text = []
    for chunk in chunks:
        text = chunk["text"].strip()
        if not re.match(
            r"\s*(foreign you|foreign|Thank you for watching!|you there|you)\s*$", text
        ):
            if text.strip() == "":
                continue
            start = chunk["start"]
            end = chunk["end"]
            pattern = r"\n(?!\d+\.\d+\s*-->)"
            text = re.sub(pattern, "", text)
            processed_chunks_text.append(text.strip())
            processed_chunks.append(f"{start:.2f} --> {end:.2f}: {text} <br>")
        else:
            break
    print(f"{processed_chunks=}\n")
    print(f"{processed_chunks_text=}\n")
    print(
        re.search(r"\s*foreign\s*$", processed_chunks_text[-1])
        if processed_chunks_text
        else None
    )

    if processed_chunks_text and re.search(
        r"\s*foreign\s*$", processed_chunks_text[-1]
    ):
        processed_chunks_text[-1] = re.sub(
            r"\s*foreign\s*$", "", processed_chunks_text[-1]
        )
        processed_chunks[-1] = re.sub(r"foreign\s*<br>", "<br>", processed_chunks[-1])
    return "\n".join(processed_chunks), " ".join(processed_chunks_text)


def hf_process_chunks(chunks):
    processed_chunks = []
    processed_chunks_text = []
    for chunk in chunks:
        text = chunk["text"].strip()
        if not re.match(r"(foreign you|foreign|you there|you)\s*$", text):
            if text.strip() == "":
                continue
            start = chunk["timestamp"][0]
            end = chunk["timestamp"][1]
            pattern = r"\n(?!\d+\.\d+\s*-->)"
            text = re.sub(pattern, "", text)
            processed_chunks_text.append(text.strip())
            processed_chunks.append(f"{start:.2f} --> {end:.2f}: {text.strip()} <br>")
        else:
            break
    print(f"{processed_chunks=}\n")
    print(f"{processed_chunks_text=}\n")
    print(
        re.search(r"\s*foreign\s*$", processed_chunks_text[-1])
        if processed_chunks_text
        else None
    )

    if processed_chunks_text and re.search(
        r"\s*foreign\s*$", processed_chunks_text[-1]
    ):
        processed_chunks_text[-1] = re.sub(
            r"\s*foreign\s*$", "", processed_chunks_text[-1]
        )
        processed_chunks[-1] = re.sub(r"foreign\s*<br>", "<br>", processed_chunks[-1])
    return "\n".join(processed_chunks), " ".join(processed_chunks_text)


def hf_seq_process_chunks(chunks):
    processed_chunks = []
    processed_chunks_text = []
    delta_time = 0.0
    global_start = chunks[0]["timestamp"][0]
    prev_end = -1.0
    prev_dur = 0.0
    accumulate_ts = False
    for chunk in chunks:
        text = chunk["text"].strip()
        if not re.match(r"\s*(foreign you|foreign|you there|you)\s*$", text):
            if text.strip() == "":
                continue
            start = chunk["timestamp"][0]
            if start < prev_end:
                accumulate_ts = True
            end = chunk["timestamp"][1]
            if start < prev_end:
                prev_dur += delta_time
            # print(f"{prev_dur=}")

            delta_time = end - global_start
            # print(f"{delta_time=}")

            prev_end = end
            # print(f"{prev_end=}")
            if accumulate_ts:
                start += prev_dur
            if accumulate_ts:
                end += prev_dur
            # print(f"{start=}, {end=}, {prev_dur=}")

            pattern = r"\n(?!\d+\.\d+\s*-->)"
            text = re.sub(pattern, "", text)
            processed_chunks_text.append(text.strip())
            processed_chunks.append(f"{start:.2f} --> {end:.2f}: {text.strip()} <br>")
        else:
            break
    print(f"{processed_chunks=}\n")
    print(f"{processed_chunks_text=}\n")
    print(
        re.search(r"\s*foreign\s*$", processed_chunks_text[-1])
        if processed_chunks_text
        else None
    )

    if processed_chunks_text and re.search(
        r"\s*foreign\s*$", processed_chunks_text[-1]
    ):
        processed_chunks_text[-1] = re.sub(
            r"\s*foreign\s*$", "", processed_chunks_text[-1]
        )
        processed_chunks[-1] = re.sub(r"foreign\s*<br>", "<br>", processed_chunks[-1])
    return "\n".join(processed_chunks), " ".join(processed_chunks_text)


original_timestamp_html = """
    <div style="background: white; border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; width: 100%; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); flex: 1; margin-right: 10px;">
        <div style="color: #374151; font-size: 14px; font-weight: 500; margin-bottom: 8px;">Timestamp Text</div>
        <div id="timestampText"; style="color: #6b7280; font-size: 14px; line-height: 1.5; min-height: 100px; font-family: system-ui, sans-serif;"></div>
    </div>
    """

original_transcription_html = """
    <div style="background: white; border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; width: 100%; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); flex: 1; margin-right: 10px;">
        <div style="color: #374151; font-size: 14px; font-weight: 500; margin-bottom: 8px;">Transcription Text</div>
        <div id="transcriptionText"; style="color: #6b7280; font-size: 14px; line-height: 1.5; min-height: 100px; font-family: system-ui, sans-serif;"></div>
    </div>
    """


def reset():
    return original_timestamp_html, original_transcription_html


event_process_js = """
<script>
function getTime() {
    lastIndex = -1;
    setInterval(() => {
        time = document.getElementById('time');
        timestampText = document.getElementById('timestampText');
        if(timestampText && timestampText.innerText != '') {
            if(time == null) {
                timestampText.innerText = '';
                transcriptionText = document.getElementById('transcriptionText');
                if(transcriptionText) {
                    transcriptionText.innerText = '';
                }
                lastIndex = -1;
                return; 
            }
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
        else {
            lastIndex = -1;
        }
    }, 50);
}
setTimeout(getTime, 1000);
</script>
"""
demo = gr.Blocks(
    head=event_process_js,
    theme=gr.themes.Default(primary_hue="emerald", secondary_hue="green"),
)
with demo:
    audio = gr.Audio(sources=["upload", "microphone"], type="filepath")
    inf_strategy = gr.Dropdown(
        label="Inference Strategy",
        choices=[
            "HuggingFace Chunking",
            "HuggingFace Sequential",
            "OLMoASR Sequential",
        ],
        value="HuggingFace Chunking",
        multiselect=False,
        info="Select the inference strategy for transcription.",
        elem_id="inf_strategy",
    )
    main_transcribe_button = gr.Button(
        "Transcribe",
        variant="primary",
    )
    with gr.Row():
        timestampText = gr.HTML(original_timestamp_html)

        transcriptionText = gr.HTML(original_transcription_html)
    inf_strategy.change(
        fn=reset,
        inputs=[],
        outputs=[timestampText, transcriptionText],
    )
    main_transcribe_button.click(
        fn=main_transcribe,
        inputs=[inf_strategy, audio, timestampText, transcriptionText],
        outputs=[timestampText, transcriptionText],
    )
demo.launch(share=True)
