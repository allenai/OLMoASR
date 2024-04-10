from open_whisper.utils import calculate_wer

def test_calculate_wer():
    pair = ("hello world", "hello world")
    assert calculate_wer(pair) == 0.0
    pair = ("hello world", "hello")
    assert calculate_wer(pair) == 0.5
    pair = ("", "hello world")
    assert calculate_wer(pair) == 0.0