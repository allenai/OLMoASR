"""Main entry point for the olmoasr package."""
import io
import os
from typing import Optional, Union
import torch
from olmoasr import (
    model,
    inf_model,
    preprocess,
    utils,
)
# from whisper import audio, decoding, transcribe
from whisper import audio, decoding
from olmoasr import transcribe
from olmoasr.model import ModelDimensions, OLMoASR
from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim


# should add more features (loading in model checkpoints by identifiers with dictionary of checkpoint paths)
def load_model(
    name: str,
    device: Optional[Union[str, torch.device]] = None,
    inference: bool = False,
    in_memory: bool = False,
) -> OLMoASR:
    """
    Load a Whisper ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"
    in_memory: bool
        whether to preload the model weights into host memory

    Returns
    -------
    model : OLMoASR
        The OLMoASR model instance
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.isfile(name):
        checkpoint_file = open(name, "rb").read() if in_memory else name
        alignment_heads = None

    with (
        io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        checkpoint = torch.load(fp, map_location=device, weights_only=False)
    del checkpoint_file

    dims = ModelDimensions(**checkpoint["dims"])
    if inference:
        model = inf_model.OLMoASR(dims)
    else:
        model = model.OLMoASR(dims)
    model.load_state_dict(checkpoint["model_state_dict"])

    if alignment_heads is not None:
        model.set_alignment_heads(alignment_heads)

    return model.to(device)
