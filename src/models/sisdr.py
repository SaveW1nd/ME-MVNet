"""SI-SDR utilities."""

from __future__ import annotations

import torch


def _flatten_wave(x: torch.Tensor) -> torch.Tensor:
    """Flatten IQ waveform to shape (B, T)."""
    if x.ndim == 3:
        # (B, 2, N) -> (B, 2N)
        return x.reshape(x.shape[0], -1)
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected tensor rank 2/3, got shape {tuple(x.shape)}")


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-Invariant SDR for batched waveforms.

    Args:
        est: Estimated waveform, shape (B,2,N) or (B,T).
        ref: Reference waveform, shape (B,2,N) or (B,T).
        eps: Numerical stability.

    Returns:
        SI-SDR tensor, shape (B,).
    """
    est_f = _flatten_wave(est)
    ref_f = _flatten_wave(ref)
    if est_f.shape != ref_f.shape:
        raise ValueError(f"Shape mismatch: {tuple(est_f.shape)} vs {tuple(ref_f.shape)}")

    ref_energy = torch.sum(ref_f * ref_f, dim=1, keepdim=True) + eps
    proj = torch.sum(est_f * ref_f, dim=1, keepdim=True) * ref_f / ref_energy
    noise = est_f - proj

    ratio = (torch.sum(proj * proj, dim=1) + eps) / (torch.sum(noise * noise, dim=1) + eps)
    return 10.0 * torch.log10(ratio + eps)
