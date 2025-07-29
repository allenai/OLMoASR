from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from olmoasr.config.model_dims import ModelDimensions

from whisper.decoding import decode as decode_function
from whisper.decoding import detect_language as detect_language_function
# from whisper.transcribe import transcribe as transcribe_function
from olmoasr.transcribe import transcribe as transcribe_function

class LayerNorm(nn.LayerNorm):
    """
    This function is from OpenAI's Whisper repository.
    The original version can be found at: https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/model.py#L30
    References:
        - Author: OpenAI
        - Source: Whisper GitHub Repository
        - License: MIT License
        - Date of Access: Novemeber 10, 2024
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through layer normalization.
        
        Parameters
        ----------
        x : Tensor
            Input tensor to normalize
            
        Returns
        -------
        Tensor
            Normalized tensor with same shape as input
        """
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    """
    This class is based on an implementation by OpenAI from the Whisper repository.
    The original version can be found at: https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/model.py#L35
    References:
        - Author: OpenAI
        - Source: Whisper GitHub Repository
        - License: MIT License
        - Date of Access: Novemeber 10, 2024
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Initialize Linear layer with Kaiming normal initialization.
        
        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        bias : bool, optional
            Whether to include bias term, by default True
        device : optional
            Device to place the layer on
        dtype : optional
            Data type for the layer parameters
        """
        super(Linear, self).__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )

        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through linear layer with dtype casting.
        
        Parameters
        ----------
        x : Tensor
            Input tensor
            
        Returns
        -------
        Tensor
            Output tensor after linear transformation
        """
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    """
    This class is based on an implementation by OpenAI from the Whisper repository.
    The original version can be found at: https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/model.py#L44
    References:
        - Author: OpenAI
        - Source: Whisper GitHub Repository
        - License: MIT License
        - Date of Access: Novemeber 10, 2024
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ):
        """
        Initialize 1D convolution layer with Kaiming normal initialization.
        
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Size of the convolving kernel
        stride : int, optional
            Stride of the convolution, by default 1
        padding : int, optional
            Padding added to input, by default 0
        dilation : int, optional
            Spacing between kernel elements, by default 1
        groups : int, optional
            Number of blocked connections, by default 1
        bias : bool, optional
            Whether to include bias term, by default True
        padding_mode : str, optional
            Padding mode, by default "zeros"
        device : optional
            Device to place the layer on
        dtype : optional
            Data type for the layer parameters
        """
        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        """
        Forward convolution with dtype casting.
        
        Parameters
        ----------
        x : Tensor
            Input tensor
        weight : Tensor
            Weight tensor
        bias : Optional[Tensor]
            Bias tensor
            
        Returns
        -------
        Tensor
            Output tensor after convolution
        """
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


# don't really understand the positional embeddings and the intuition behind it
def sinusoids(length, channels, max_timescale=10000):
    """
    This function is from OpenAI's Whisper repository.
    The original version can be found at: https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/model.py#L53
    References:
        - Author: OpenAI
        - Source: Whisper GitHub Repository
        - License: MIT License
        - Date of Access: Novemeber 10, 2024
    """
    """
    Returns sinusoids for positional embedding.
    
    Parameters
    ----------
    length : int
        Sequence length
    channels : int
        Number of channels (must be even)
    max_timescale : int, optional
        Maximum timescale for sinusoids, by default 10000
        
    Returns
    -------
    torch.Tensor
        Sinusoidal positional embeddings of shape (length, channels)
    """
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    """
    This class is from OpenAI's Whisper repository.
    The original version can be found at: https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/model.py#L62
    References:
        - Author: OpenAI
        - Source: Whisper GitHub Repository
        - License: MIT License
        - Date of Access: Novemeber 10, 2024
    """

    def __init__(self, n_state: int, n_head: int):
        """
        Initialize multi-head attention layer.
        
        Parameters
        ----------
        n_state : int
            Dimension of the model
        n_head : int
            Number of attention heads
        """
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)  # W_q
        nn.init.kaiming_normal_(self.query.weight, mode="fan_in", nonlinearity="relu")
        self.key = Linear(n_state, n_state, bias=False)  # W_k
        nn.init.kaiming_normal_(self.key.weight, mode="fan_in", nonlinearity="relu")
        self.value = Linear(n_state, n_state)  # W_v
        nn.init.kaiming_normal_(self.value.weight, mode="fan_in", nonlinearity="relu")
        self.out = Linear(n_state, n_state)  # W_o
        nn.init.kaiming_normal_(self.out.weight, mode="fan_in", nonlinearity="relu")

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        verbose: bool = False,
    ):
        """
        Forward pass through multi-head attention.
        
        Parameters
        ----------
        x : Tensor
            Input tensor for query computation
        xa : Optional[Tensor], optional
            Input tensor for key/value computation (for cross-attention), by default None
        mask : Optional[Tensor], optional
            Attention mask, by default None
        kv_cache : Optional[dict], optional
            Key-value cache for efficient inference, by default None
        verbose : bool, optional
            Whether to print debug information, by default False
            
        Returns
        -------
        Tuple[Tensor, Optional[Tensor]]
            Output tensor and attention weights (if computed)
        """
        q = self.query(x)  # W_q * x
        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        if verbose:
            print("MultiHeadAttention forward pass")
            print(f"{q=}")
            print(f"{k=}")
            print(f"{v=}")
            print("Attention computation")

        qk = None
        # for eval loop during training
        inference = False

        if mask is not None:
            # for eval loop during training, no padding mask, just causal mask
            if len(mask.shape) == 2:
                inference = True
            # for training loop, padding mask + causal mask
            else:
                mask = mask.unsqueeze(dim=1)
        
        # TODO: debug why SDPA is not working for inference
        if inference:
            wv, qk = self.qkv_attention(q, k, v, mask, verbose=verbose)
        # TODO: expand support for regular QKV attention when SDPA is not possible
        # for training, use SDPA
        else:
            q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
            wv = (
                F.scaled_dot_product_attention(
                    query=q, key=k, value=v, attn_mask=mask
                )
                .permute(0, 2, 1, 3)
                .flatten(start_dim=2)
            )

        if verbose:
            print(f"{wv=}")

        return self.out(wv), qk

    def qkv_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
        verbose: bool = False,
    ):
        """
        Compute attention using manual Q, K, V computation.
        
        Parameters
        ----------
        q : Tensor
            Query tensor
        k : Tensor
            Key tensor
        v : Tensor
            Value tensor
        mask : Optional[Tensor], optional
            Attention mask, by default None
        verbose : bool, optional
            Whether to print debug information, by default False
            
        Returns
        -------
        Tuple[Tensor, Tensor]
            Weighted values and attention weights
        """
        *_, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25

        if verbose:
            print(f"{scale=}")

        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if verbose:
            print("Scaling QKV")
            print(f"{q.shape=}")
            print(f"{torch.max(q, dim=-1).values=}")
            print(f"{torch.min(q, dim=-1).values=}")
            print(f"{q=}")
            print(f"{k.shape=}")
            print(f"{torch.max(k, dim=-1).values=}")
            print(f"{torch.min(k, dim=-1).values=}")
            print(f"{k=}")
            print(f"{v.shape=}")
            print(f"{torch.max(v, dim=-1).values=}")
            print(f"{torch.min(v, dim=-1).values=}")
            print(f"{v=}")

        qk = q @ k

        if verbose:
            if mask is not None:
                print("Before adding mask")
            print(f"{qk.shape=}")
            print(f"{torch.max(qk, dim=-1).values=}")
            print(f"{torch.min(qk, dim=-1).values=}")
            print(f"{qk=}")

        if mask is not None:
            qk = qk + mask

        if verbose:
            if mask is not None:
                print("After adding mask")
            print(f"{qk.shape=}")
            print(f"{torch.max(qk, dim=-1).values=}")
            print(f"{torch.min(qk, dim=-1).values=}")
            print(f"{qk=}")

        # float16 -> float32 for softmax
        qk = qk.float()

        if verbose:
            print("Converting QK to torch.float32")
            print(f"{qk.shape=}")
            print(f"{torch.max(qk, dim=-1).values=}")
            print(f"{torch.min(qk, dim=-1).values=}")
            print(f"{qk=}")

        # float32 -> float16 after softmax
        w = F.softmax(qk, dim=-1).to(q.dtype)

        if verbose:
            print("Softmax")
            print(f"{torch.max(w, dim=-1).values=}")
            print(f"{torch.min(w, dim=-1).values=}")
            print(f"{w.shape=}")
            print(f"{w=}")

        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    """
    This class is from OpenAI's Whisper repository.
    The original version can be found at: https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/model.py#L111
    References:
        - Author: OpenAI
        - Source: Whisper GitHub Repository
        - License: MIT License
        - Date of Access: Novemeber 10, 2024
    """

    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        """
        Initialize residual attention block.
        
        Parameters
        ----------
        n_state : int
            Dimension of the model
        n_head : int
            Number of attention heads
        cross_attention : bool, optional
            Whether to include cross-attention, by default False
        """
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        verbose: bool = False,
    ):
        """
        Forward pass through residual attention block.
        
        Parameters
        ----------
        x : Tensor
            Input tensor
        xa : Optional[Tensor], optional
            Cross-attention input tensor, by default None
        mask : Optional[Tensor], optional
            Attention mask, by default None
        kv_cache : Optional[dict], optional
            Key-value cache for efficient inference, by default None
        verbose : bool, optional
            Whether to print debug information, by default False
            
        Returns
        -------
        Tensor
            Output tensor after residual attention block
        """
        x = (
            x
            + self.attn(
                self.attn_ln(x), mask=mask, kv_cache=kv_cache
            )[0]
        )
        if self.cross_attn:
            x = (
                x
                + self.cross_attn(
                    self.cross_attn_ln(x), xa, kv_cache=kv_cache, verbose=verbose
                )[0]
            )
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    """
    This class is from OpenAI's Whisper repository.
    The original version can be found at: https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/model.py#L143
    References:
        - Author: OpenAI
        - Source: Whisper GitHub Repository
        - License: MIT License
        - Date of Access: Novemeber 10, 2024
    """

    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        """
        Initialize audio encoder.
        
        Parameters
        ----------
        n_mels : int
            Number of mel-frequency channels
        n_ctx : int
            Context length
        n_state : int
            Dimension of the model
        n_head : int
            Number of attention heads
        n_layer : int
            Number of layers
        """
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor, verbose: bool = False):
        """
        Forward pass through audio encoder.
        
        Parameters
        ----------
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            The mel spectrogram of the audio
        verbose : bool, optional
            Whether to print debug information, by default False
            
        Returns
        -------
        Tensor
            Encoded audio features
        """
        if verbose:
            print("Starting audio encoder forward pass")
            print(f"{x.shape=}")
            print(f"{x=}")

        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        if verbose:
            print("After convolution")
            print(f"{x.shape=}")
            print(f"{x=}")

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        if verbose:
            print("After adding positional embeddings")
            print(f"{x.shape=}")
            print(f"{x=}")

        for block in self.blocks:
            x = block(x, verbose=verbose)

            if verbose:
                print(f"{x.shape=}")
                print(f"{x=}")

        x = self.ln_post(x)

        if verbose:
            print("After layer norm")
            print(f"{x.shape=}")
            print(f"{x=}")

        return x


class TextDecoder(nn.Module):
    """
    This class is based on an implementation by OpenAI from the Whisper repository.
    Modifications were made to work with the training code.

    The original version can be found at: https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/model.py#L176
    References:
        - Author: OpenAI
        - Source: Whisper GitHub Repository
        - License: MIT License
        - Date of Access: Novemeber 10, 2024
    """

    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        """
        Initialize text decoder.
        
        Parameters
        ----------
        n_vocab : int
            Vocabulary size
        n_ctx : int
            Context length
        n_state : int
            Dimension of the model
        n_head : int
            Number of attention heads
        n_layer : int
            Number of layers
        """
        super().__init__()

        # vocab size + 1 for padding token (51864)
        # for inference, use scripts/eval/gen_inf_ckpt.py to generate the checkpoint w/out padding token
        # since inference script doesn't use padding token
        # this was stupid in hindsight (I'm more knowledgeable now about training), but I'm going to keep it to not mess with anything

        self.token_embedding = nn.Embedding(
            n_vocab + 1, n_state, padding_idx=51864 if n_vocab == 51864 else 51865
        )
        nn.init.kaiming_normal_(
            self.token_embedding.weight, mode="fan_in", nonlinearity="relu"
        )

        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        nn.init.kaiming_normal_(
            self.positional_embedding, mode="fan_in", nonlinearity="relu"
        )

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)
        # causal mask to prevent attention to future tokens
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        kv_cache: Optional[dict] = None,
        padding_mask: Optional[Tensor] = None,
        verbose: bool = False,
    ):
        """
        Forward pass through text decoder.
        
        Parameters
        ----------
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            The text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            The encoded audio features to be attended on
        kv_cache : Optional[dict], optional
            Key-value cache for efficient inference, by default None
        padding_mask : Optional[Tensor], optional
            Mask for padding tokens, by default None
        verbose : bool, optional
            Whether to print debug information, by default False
            
        Returns
        -------
        Tensor
            Logits over vocabulary
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0

        if verbose:
            print("Starting text decoder forward pass")
            print(f"{x.shape=}")
            print(f"{x=}")
            print(f"{xa.shape=}")
            print(f"{xa=}")
            print(f"{self.token_embedding(x)=}")
            print(f"{self.positional_embedding[offset : offset + x.shape[-1]]=}")

        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        if verbose:
            print("After token and positional embedding")
            print(f"{x.shape=}")
            print(f"{x=}")

        n_ctx = x.shape[1]
        if padding_mask is not None:
            full_mask = padding_mask + self.mask
        else:
            full_mask = self.mask[:n_ctx, :n_ctx]

        if verbose:
            print("Starting decoder blocks")

        for block in self.blocks:
            x = block(
                x,
                xa,
                mask=full_mask,
                kv_cache=kv_cache,
                verbose=verbose,
            )

            if verbose:
                print(f"{x.shape=}")
                print(f"{x=}")

        x = self.ln(x)
        
        if verbose:
            print("After layer norm")
            print(f"{x.shape=}")
            print(f"{x=}")

        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        if verbose:
            print(f"{logits=}")

        return logits


class OLMoASR(nn.Module):
    """
    This class is from OpenAI's Whisper repository.
    The original version can be found at: https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/model.py#L221
    References:
        - Author: OpenAI
        - Source: Whisper GitHub Repository
        - License: MIT License
        - Date of Access: Novemeber 10, 2024
    """

    def __init__(self, dims: ModelDimensions):
        """
        Initialize OLMoASR model.
        
        Parameters
        ----------
        dims : ModelDimensions
            Model dimensions configuration
        """
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,  # mel channels
            self.dims.n_audio_ctx,  # context length of audio embedding
            self.dims.n_audio_state,  # dimension of audio embedding
            self.dims.n_audio_head,  # number of heads in encoder
            self.dims.n_audio_layer,  # number of layers in encoder
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,  # vocab size
            self.dims.n_text_ctx,  # context length of text embedding
            self.dims.n_text_state,  # dimension of text embedding
            self.dims.n_text_head,  # number of heads in decoder
            self.dims.n_text_layer,  # number of layers in decoder
        )

    def embed_audio(self, mel: torch.Tensor):
        """
        Embed audio features using the encoder.
        
        Parameters
        ----------
        mel : torch.Tensor
            Mel spectrogram input
            
        Returns
        -------
        torch.Tensor
            Encoded audio features
        """
        return self.encoder(mel)

    def logits(
        self,
        tokens: torch.Tensor,
        audio_features: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        """
        Compute logits for given tokens and audio features.
        
        Parameters
        ----------
        tokens : torch.Tensor
            Input tokens
        audio_features : torch.Tensor
            Encoded audio features
        padding_mask : torch.Tensor, optional
            Mask for padding tokens, by default None
            
        Returns
        -------
        torch.Tensor
            Logits over vocabulary
        """
        return self.decoder(tokens, audio_features, padding_mask=padding_mask)

    def forward(
        self,
        mel: torch.Tensor,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor = None,
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Parameters
        ----------
        mel : torch.Tensor
            Mel spectrogram input
        tokens : torch.Tensor
            Input tokens
        padding_mask : torch.Tensor, optional
            Mask for padding tokens, by default None
        verbose : bool, optional
            Whether to print debug information, by default False
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing model outputs
        """
        return self.decoder(
            tokens,
            self.encoder(mel, verbose=verbose),
            padding_mask=padding_mask,
            verbose=verbose,
        )

    @property
    def device(self):
        """
        Get the device of the model.
        
        Returns
        -------
        torch.device
            Device where the model parameters are located
        """
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        """
        Check if the model is multilingual.
        
        Returns
        -------
        bool
            True if model supports multiple languages
        """
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        """
        Get the number of languages supported by the model.
        
        Returns
        -------
        int
            Number of supported languages
        """
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        Install hooks for key-value caching to enable efficient inference.
        
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Parameters
        ----------
        cache : Optional[dict], optional
            Existing cache dictionary to extend, by default None

        Returns
        -------
        Tuple[Dict[nn.Module, torch.Tensor], List[RemovableHandle]]
            cache : Dict[nn.Module, torch.Tensor]
                A dictionary object mapping the key/value projection modules to its cache
            hooks : List[RemovableHandle]
                List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
