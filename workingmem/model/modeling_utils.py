
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard learned or sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, learnable: bool = True):
        super().__init__()
        self.learnable = learnable

        if learnable:
            # Learned positional embeddings
            self.pos_embedding = nn.Embedding(max_len, d_model)
        else:
            # Sinusoidal positional encoding
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Positional encodings of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        if self.learnable:
            positions = (
                torch.arange(seq_len, device=x.device)
                .unsqueeze(0)
                .expand(x.size(0), -1)
            )
            return self.pos_embedding(positions)
        else:
            return self.pe[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as described in https://arxiv.org/abs/2104.09864

    Note: This implementation applies rotary embeddings to the full sequence embeddings.
    For true RoPE behavior in multi-head attention, custom attention layers would be needed.
    This implementation approximates RoPE by rotating the embeddings before passing to the transformer.
    """

    def __init__(self, dim: int, max_len: int = 5000, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base

        # Precompute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for rotary embeddings
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device):
        """Update the cached cos/sin values if sequence length changed."""
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Repeat frequencies for full dimension
            emb = torch.cat((freqs, freqs), dim=-1)
            # Add batch and head dimensions for broadcasting
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embedding to input tensor.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with rotary embeddings applied, shape (batch_size, seq_len, d_model)
        """
        seq_len = x.shape[1]
        self._update_cache(seq_len, x.device)

        # Apply rotary embedding
        # Shape: (seq_len, d_model)
        cos = self._cos_cached[:seq_len, : x.shape[-1]]
        sin = self._sin_cached[:seq_len, : x.shape[-1]]

        # Broadcast to batch dimension
        cos = cos.unsqueeze(0)  # (1, seq_len, d_model)
        sin = sin.unsqueeze(0)  # (1, seq_len, d_model)

        # Apply rotation
        x_rotated = (x * cos) + (self.rotate_half(x) * sin)

        return x_rotated
