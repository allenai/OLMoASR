from dataclasses import dataclass


@dataclass
class ModelDimensions:
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


variant_to_dims = {
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
}
