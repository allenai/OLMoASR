from open_whisper.utils import TranscriptReader


class TestReadSRT:
    def test_read_srt():
        file_path = "tests/data/transcript/2CWmrpsN41E.en.srt"
        transcript, transcript_start, transcript_end = TranscriptReader(
            file_path=file_path
        ).read()
        assert len(transcript.keys()) == 20
        assert transcript_start == "00:00:05.360"
        assert transcript_end == "00:03:35.520"

    def test_read_SRT_empty():
        file_path = "tests/data/empty_transcript.srt"
        transcript, transcript_start, transcript_end = TranscriptReader(
            file_path=file_path
        ).read()

        assert len(transcript.keys()) == 0
        assert transcript_start == ""
        assert transcript_end == ""
