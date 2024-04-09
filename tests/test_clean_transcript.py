from open_whisper.utils import clean_transcript, CHARS_TO_REMOVE
import pysrt
import pytest


class TestCleanTranscript:
    def test_clean_transcript(self):
        file_path = "tests/data/transcript/2CWmrpsN41E.en.srt"
        cleaned = clean_transcript(file_path=file_path)
        assert cleaned is True
        subs = pysrt.open(file_path)
        for sub in subs:
            assert all([char not in sub.text for char in CHARS_TO_REMOVE])

    def test_clean_transcript_empty(self):
        file_path = "tests/data/empty_transcript.srt"
        cleaned = clean_transcript(file_path=file_path)
        assert cleaned is None
        subs = pysrt.open(file_path)
        assert len(subs) == 0
        with open(file_path, "r") as f:
            assert f.read().strip() == ""
