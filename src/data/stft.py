"""STFT helpers."""

from __future__ import annotations

import torch


def iq_to_logmag_stft(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    """Convert IQ input to log-magnitude STFT.

    Args:
        x: Tensor with shape (B, 2, N), float32.
        n_fft: FFT length.
        hop_length: STFT hop.
        win_length: Window length.

    Returns:
        Tensor with shape (B, 1, F, T), float32.
    """
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"Expected x shape (B,2,N), got {tuple(x.shape)}")

    complex_x = torch.complex(x[:, 0, :], x[:, 1, :])
    window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        input=complex_x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    mag = torch.log1p(spec.abs())
    return mag.unsqueeze(1)
