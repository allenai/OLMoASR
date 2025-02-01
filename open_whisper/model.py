from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from open_whisper.config.model_dims import ModelDimensions

from whisper.decoding import decode as decode_function
from whisper.decoding import detect_language as detect_language_function
from whisper.transcribe import transcribe as transcribe_function

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
        super(Linear, self).__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )

        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
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
    """Returns sinusoids for positional embedding"""
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
        block_count: int = 0,
    ):
        q = self.query(x)  # W_q * x
        # not sure when which branch is taken
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
            
        # wv, qk = self.qkv_attention(q, k, v, mask, verbose=verbose, block_count=block_count)
        
        qk = None
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask[:n_ctx, :n_ctx]
            else:
                mask = mask.unsqueeze(dim=1)
        wv = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask, scale=scale).permute(0, 2, 1, 3).flatten(start_dim=2)
        
        if verbose:
            print(f"{wv=}")
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, verbose: bool = False, block_count: int = 0
    ):
        n_batch, n_ctx, n_state = q.shape
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
            # if len(mask.shape) == 2:
            #     qk = qk + mask
            # else:
            #     qk = (
            #         qk
            #         + mask.unsqueeze(dim=1).repeat(1, self.n_head, 1, 1)[
            #             :, :, :n_ctx, :n_ctx
            #         ]
            #     )
        if verbose:
            if mask is not None:
                print("After adding mask")
            print(f"{qk.shape=}")
            print(f"{torch.max(qk, dim=-1).values=}")
            print(f"{torch.min(qk, dim=-1).values=}")
            print(f"{qk=}")
            
        qk = qk.float()
        if verbose:
            print("Converting QK to torch.float32")
            print(f"{qk.shape=}")
            print(f"{torch.max(qk, dim=-1).values=}")
            print(f"{torch.min(qk, dim=-1).values=}")
            print(f"{qk=}")

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
        block_count: int = 0,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache, block_count=block_count)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache, verbose=verbose)[0]
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
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
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

        block_count = 0
        for block in self.blocks:
            if verbose:
                print(f"Block {block_count}")
            x = block(x, verbose=verbose)
            if verbose:
                print(f"{x.shape=}")
                print(f"{x=}")
            block_count += 1

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
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab + 1, n_state, padding_idx=51864 if n_vocab == 51864 else 51865)
        nn.init.kaiming_normal_(
            self.token_embedding.weight, mode="fan_in", nonlinearity="relu"
        )

        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        nn.init.kaiming_normal_(self.positional_embedding, mode="fan_in", nonlinearity="relu")

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
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
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

        # n_ctx = x.shape[1]
        if padding_mask is not None:
            # mask = (
            #     torch.empty(n_ctx, n_ctx)
            #     .fill_(-np.inf)
            #     .triu_(1)
            #     .to(device=padding_mask.device)
            # )
            # self.mask = padding_mask + mask
            full_mask = padding_mask + self.mask
        else:
            full_mask = self.mask
            # self.mask = (
            #     torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1).to(device=x.device)
            # )
        if verbose:
            print("Starting decoder blocks")
        block_count = 0
        for block in self.blocks:
            if verbose:
                print(f"Block {block_count}")
            # x = block(x, xa, mask=self.mask, kv_cache=kv_cache, verbose=verbose, block_count=block_count)
            x = block(x, xa, mask=full_mask, kv_cache=kv_cache, verbose=verbose, block_count=block_count)
            if verbose:
                print(f"{x.shape=}")
                print(f"{x=}")
            block_count += 1
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


class Whisper(nn.Module):
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
        return self.encoder(mel)

    def logits(
        self,
        tokens: torch.Tensor,
        audio_features: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        return self.decoder(tokens, audio_features, padding_mask=padding_mask)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor, padding_mask: torch.Tensor = None, verbose: bool = False
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel, verbose=verbose), padding_mask=padding_mask, verbose=verbose)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
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
