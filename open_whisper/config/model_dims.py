from dataclasses import dataclass


@dataclass
class ModelDimensions:
    """
    This class is from OpenAI's Whisper repository.
    The original version can be found at: https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/model.py#L17
    References:
        - Author: OpenAI
        - Source: Whisper GitHub Repository
        - License: MIT License
        - Date of Access: Novemeber 10, 2024
    """

    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


VARIANT_TO_DIMS = {
    "tiny": ModelDimensions(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=384,
        n_audio_head=6,
        n_audio_layer=4,
        n_vocab=51864,
        n_text_ctx=448,
        n_text_state=384,
        n_text_head=6,
        n_text_layer=4,
    ),
    "base": ModelDimensions(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=512,
        n_audio_head=8,
        n_audio_layer=6,
        n_vocab=51864,
        n_text_ctx=448,
        n_text_state=512,
        n_text_head=8,
        n_text_layer=6,
    ),
    "small": ModelDimensions(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=768,
        n_audio_head=12,
        n_audio_layer=12,
        n_vocab=51864,
        n_text_ctx=448,
        n_text_state=768,
        n_text_head=12,
        n_text_layer=12,
    ),
    "medium": ModelDimensions(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=1024,
        n_audio_head=16,
        n_audio_layer=24,
        n_vocab=51864,
        n_text_ctx=448,
        n_text_state=1024,
        n_text_head=16,
        n_text_layer=24,
    ),
    "large": ModelDimensions(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=1280,
        n_audio_head=20,
        n_audio_layer=32,
        n_vocab=51864,
        n_text_ctx=448,
        n_text_state=1280,
        n_text_head=20,
        n_text_layer=32,
    ),
    "tiny_ml": ModelDimensions(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=384,
        n_audio_head=6,
        n_audio_layer=4,
        n_vocab=51865,
        n_text_ctx=448,
        n_text_state=384,
        n_text_head=6,
        n_text_layer=4,
    ),
}
