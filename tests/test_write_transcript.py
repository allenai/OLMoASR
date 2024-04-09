from open_whisper.utils import write_segment
import pysrt
from ast import literal_eval
import json


def test_write_segment():
    timestamps = [
        ("00:00:05.360", "00:00:12.720"),
        ("00:00:13.360", "00:00:17.680"),
        ("00:00:18.640", "00:00:25.920"),
        ("00:00:27.200", "00:00:34.800"),
    ]
    with open("tests/data/transcript_dict.json", "r") as f:
        d = json.load(f)

    transcript = {literal_eval(k): v for k, v in d.items()}
    output_dir = "tests/data"
    ext = "srt"

    write_segment(timestamps, transcript, output_dir, ext)

    file_path = f"{output_dir}/{timestamps[0][0]}_{timestamps[-1][1]}.{ext}"
    subs = pysrt.open(file_path)
    for i, sub in enumerate(subs):
        assert sub.start == timestamps[i][0]
        assert sub.end == timestamps[i][1]
        assert sub.text == transcript[(timestamps[i][0], timestamps[i][1])]
