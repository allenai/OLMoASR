"""Main entry point for the olmoasr package."""

import io
import os
from pathlib import Path
import urllib.request
import urllib.error
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

MODEL2LINK = {
    "tiny": "https://huggingface.co/allenai/OLMoASR/resolve/main/models/OLMoASR-tiny.en.pt",
    "base": "https://huggingface.co/allenai/OLMoASR/resolve/main/models/OLMoASR-base.en.pt",
    "small": "https://huggingface.co/allenai/OLMoASR/resolve/main/models/OLMoASR-small.en.pt",
    "medium": "https://huggingface.co/allenai/OLMoASR/resolve/main/models/OLMoASR-medium-v2.en.pt",
    "large": "https://huggingface.co/allenai/OLMoASR/resolve/main/models/OLMoASR-large.en.pt",
    "large-v2": "https://huggingface.co/allenai/OLMoASR/resolve/main/models/OLMoASR-large.en-v2.pt",
}


def _get_cache_dir(download_root: Optional[str] = None) -> Path:
    """Get the cache directory for storing downloaded models."""
    if download_root is not None:
        cache_dir = Path(download_root).expanduser().resolve()
    else:
        cache_dir = Path.home() / ".cache" / "olmoasr"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _download_model(
    url: str, model_name: str, download_root: Optional[str] = None
) -> str:
    """
    Download a model from a URL and cache it locally.

    Parameters
    ----------
    url : str
        URL to download the model from
    model_name : str
        Name of the model for caching
    download_root : str, optional
        Path to download the model files; by default, it uses "~/.cache/olmoasr"

    Returns
    -------
    str
        Path to the downloaded model file
    """
    cache_dir = _get_cache_dir(download_root)
    filename = f"OLMoASR-{model_name}.pt"
    cache_path = cache_dir / filename

    # Return cached file if it exists
    if cache_path.exists():
        print(f"Using cached model: {cache_path}")
        return str(cache_path)

    print(f"Downloading {model_name} model from {url}...")
    print(f"Saving to: {cache_path}")

    try:
        # Download with progress indication
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f"\rDownloading... {percent}%", end="", flush=True)

        urllib.request.urlretrieve(url, cache_path, reporthook=progress_hook)
        print(f"\nModel downloaded successfully: {cache_path}")
        return str(cache_path)

    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download model from {url}: {e}")
    except Exception as e:
        # Clean up partial download
        if cache_path.exists():
            cache_path.unlink()
        raise RuntimeError(f"Error downloading model: {e}")


# should add more features (loading in model checkpoints by identifiers with dictionary of checkpoint paths)
def load_model(
    name: str,
    device: Optional[Union[str, torch.device]] = None,
    download_root: Optional[str] = None,
    inference: bool = False,
    in_memory: bool = False,
) -> OLMoASR:
    """
    Load a OLMoASR model

    Parameters
    ----------
    name : str
        one of the official model names listed in MODEL2LINK, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root : str, optional
        path to download the model files; by default, it uses "~/.cache/olmoasr"
    inference : bool
        whether to load the inference version of the model
    in_memory: bool
        whether to preload the model weights into host memory

    Returns
    -------
    model : OLMoASR
        The OLMoASR model instance
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if name is a model identifier in MODEL2LINK
    if name in MODEL2LINK:
        checkpoint_file = _download_model(MODEL2LINK[name], name, download_root)
    elif os.path.isfile(name):
        checkpoint_file = name
    else:
        raise ValueError(
            f"Model '{name}' not found. Available models: {list(MODEL2LINK.keys())}"
        )

    # Load model weights into memory if requested
    if in_memory:
        with open(checkpoint_file, "rb") as f:
            checkpoint_file = f.read()

    alignment_heads = None

    with (
        io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")
    ) as fp:
        checkpoint = torch.load(fp, map_location=device, weights_only=False)

    # Clean up if we loaded into memory
    if in_memory:
        del checkpoint_file

    dims = ModelDimensions(**checkpoint["dims"])
    if inference:
        model_instance = inf_model.OLMoASR(dims)
    else:
        model_instance = model.OLMoASR(dims)
    model_instance.load_state_dict(checkpoint["model_state_dict"])

    if alignment_heads is not None:
        model_instance.set_alignment_heads(alignment_heads)

    return model_instance.to(device)
