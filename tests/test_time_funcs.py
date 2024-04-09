from open_whisper.utils import (
    convert_to_milliseconds,
    calculate_difference,
    adjust_timestamp,
)
import pytest


class TestTimeFunctions:
    def test_convert_to_milliseconds(self):
        assert convert_to_milliseconds("0:0:0.0") == 0
        assert convert_to_milliseconds("0:0:1.0") == 1000
        assert convert_to_milliseconds("0:1:0.0") == 60000
        assert convert_to_milliseconds("1:0:0.0") == 3600000

    def test_calculate_difference(self):
        assert calculate_difference("00:00:05.360", "00:00:12.720") == 7360
        assert calculate_difference("00:00:05.360", "00:00:05.360") == 0
        with pytest.raises("ValueError"):
            calculate_difference("00:00:05.360", "00:00:05.320")

    def test_adjust_timestamp(self):
        assert adjust_timestamp("00:00:05.360", 2000) == "00:00:07.360"
