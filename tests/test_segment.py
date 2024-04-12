from open_whisper.preprocess import chunk_audio_transcript
import os


def test_chunk_audio_transcript():
    transcript_file = "tests/data/transcript/2CWmrpsN41E.en.srt"
    audio_file = "tests/data/audio/2CWmrpsN41E.m4a"
    chunk_audio_transcript(
        transcript_file=transcript_file,
        audio_file=audio_file,
    )
    segment_names = [
        "00:00:05.360_00:00:34.800",
        "00:00:34.800_00:00:55.200",
        "00:00:55.200_00:01:23.120",
        "00:01:23.120_00:01:24.080",
        "00:01:24.080_00:01:48.640",
        "00:01:48.640_00:02:09.120",
        "00:02:09.120_00:02:14.720",
        "00:02:14.720_00:02:35.840",
        "00:02:35.840_00:03:05.840",
        "00:03:05.840_00:03:35.520",
        "00:03:35.520_00:03:35.600",
    ]

    audio_segments = os.listdir("tests/data/audio/segments")
    transcript_segments = os.listdir("tests/data/transcript/segments")

    assert len(audio_segments) == 11
    assert len(audio_segments) == len(transcript_segments)
    assert all(
        [segment.split(".m4a")[0] in segment_names for segment in audio_segments]
    )
    assert all(
        [segment.split(".srt")[0] in segment_names for segment in transcript_segments]
    )
    assert os.path.exists(transcript_file) is False
    assert os.path.exists(audio_file) is False

if __name__ == "__main__":
    test_chunk_audio_transcript()
