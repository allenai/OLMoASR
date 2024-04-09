from open_whisper.utils import trim_audio
from pydub import AudioSegment
import librosa


def test_trim_audio():
    audio_file = "tests/data/audio/2CWmrpsN41E.m4a"
    ref_audio_file = "tests/data/00:00:05.360_00:00:34.800_ref.m4a"
    start = "00:00:05.360"
    end = "00:00:34.800"
    output_dir = "tests/data"

    trim_audio(audio_file, start, end, output_dir)

    file_path = f"{output_dir}/{start}_{end}.{audio_file.split('.')[-1]}"
    audio = AudioSegment.from_file(file_path)
    ref_audio = AudioSegment.from_file(ref_audio_file)
    audio_dur = len(audio)
    ref_audio_dur = len(ref_audio)

    assert audio_dur == ref_audio_dur
    assert librosa.load(file_path)[0] == librosa.load(ref_audio_file)[0]
