# %%
from speechbrain.inference.classifiers import EncoderClassifier
from pydub import AudioSegment
import os
import tempfile
import pycld2 as cld2
from open_whisper.utils import TranscriptReader
from typing import Tuple

# %%
model = EncoderClassifier.from_hparams("speechbrain/lang-id-voxlingua107-ecapa", savedir="checkpoints/misc/speechbrain_lang_id")


# %%
def pred_lang(audio_transcript_pair: Tuple):
    audio_file, transcript_file = audio_transcript_pair
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define the output WAV file path
        wav_path = os.path.join(temp_dir, "temp_audio.wav")

        # Load the M4A file
        audio = AudioSegment.from_file(audio_file, format="m4a")

        # Convert it to WAV and save to the temporary directory
        audio.export(wav_path, format="wav")

        signal = model.load_audio(wav_path)
        pred = model.classify_batch(signal)
        audio_details = pred[3]

    audio_lang = ""
    if len(audio_details) > 1:
        with open("logs/data/filtering/multiple_lang_audio_lang_id.txt", "a") as f:
            f.write(f"{audio_file}\t{audio_details}\n")
        return None
    else:
        audio_lang = audio_details[0].split(":")[0]

    reader = TranscriptReader(transcript_file)
    t_dict, *_ = reader.read()
    text = reader.extract_text(t_dict)
    is_reliable, _, details, vectors = cld2.detect(text, returnVectors=True)

    text_lang = ""
    if not is_reliable:
        with open("logs/data/filtering/not_reliable_text_lang_id.txt", "a") as f:
            f.write(f"{transcript_file}\t{details}\n")
        return None
    if len(vectors) > 1:
        with open("logs/data/filtering/multiple_lang_text_lang_id.txt", "a") as f:
            f.write(f"{transcript_file}\t{vectors}\n")
        return None
    else:
        text_lang = details[0][1]

    if audio_lang != "en":
        with open("logs/data/filtering/not_english_audio_data.txt", "a") as f:
            f.write(f"{audio_file}\t{audio_details}\n")
    if text_lang != "en":
        with open("logs/data/filtering/not_english_text_data.txt", "a") as f:
            f.write(f"{transcript_file}\t{details}\n")

    if audio_lang != text_lang:
        with open("logs/data/filtering/audio_text_lang_mismatch.txt", "a") as f:
            f.write(f"{audio_file}\t{audio_details}\t{transcript_file}\t{details}\n")
    else:
        with open("logs/data/filtering/audio_text_lang_match.txt", "a") as f:
            f.write(f"{audio_file}\t{audio_lang}\t{transcript_file}\t{text_lang}\n")

    return (audio_file, audio_lang, transcript_file, text_lang)


# %%
pred_lang(
    audio_transcript_pair=(
        "data/audio/__atqm8mg9E/segments/00:00:03.900_00:00:33.320.m4a",
        "data/transcripts/__atqm8mg9E/segments/00:00:03.900_00:00:33.320.srt",
    )
)
# %%
audio_transcript_pairs = []
for root, dirs, files in os.walk("data/audio"):
    if "segments" in root:
        for f in os.listdir(root):
            audio_transcript_pairs.append(
                (
                    os.path.join(root, f),
                    os.path.join(
                        root.replace("audio", "transcripts"), f.replace("m4a", "srt")
                    ),
                )
            )
audio_transcript_pairs
#%%
from tqdm import tqdm
results = []
for audio_transript_pair in tqdm(audio_transcript_pairs):
    results.append(pred_lang(audio_transript_pair))

# %%
