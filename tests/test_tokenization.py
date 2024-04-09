from scripts import AudioTextDataset
import pytest
import torch


def test_preprocess_text():
    audio_files = ["tests/data/00:00:05.360_00:00:34.800_ref.m4a"]
    transcript_files = ["tests/data/00:00:05.360_00:00:34.800_ref.srt"]

    audio_text_dataset = AudioTextDataset(
        audio_files=audio_files, transcript_files=transcript_files, n_text_ctx=448
    )

    *_, text_input, text_y, padding_mask = audio_text_dataset.__getitem__(0)

    assert text_input.shape == (448,)
    assert text_y.shape == (448,)
    assert padding_mask.shape == (448, 448)

    tokenized_text_ref = torch.load("tests/data/tokenized_text.pt")
    assert torch.equal(tokenized_text_ref["text_input"], text_input)
    assert torch.equal(tokenized_text_ref["text_y"], text_y)
    assert torch.equal(tokenized_text_ref["padding_mask"], padding_mask)
