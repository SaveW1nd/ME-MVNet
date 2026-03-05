"""CISRJ-SN style separator.

This implementation follows the paper's core design intent:
- 1D conv encoder (k=16, s=8 in default repro config)
- stacked hybrid attention blocks (local conv + gated attention)
- sigmoid mask estimation for multiple sources

To keep compatibility with the existing ME-MVSepPE pipeline, the module
returns the same keys as SepNet:
    j_hat: (B,3,2,N)
    b_hat: (B,2,N)
    masks: (B,S,1,N)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SinusoidalPositionalEncoding(nn.Module):
    """Deterministic sinusoidal positional encoding on latent time axis."""

    def __init__(self, channels: int, max_len: int = 4096) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        pe = torch.zeros(channels, max_len, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, channels, 2, dtype=torch.float32) * (-math.log(10000.0) / channels))
        pe[0::2, :] = torch.sin(pos * div).transpose(0, 1)
        if channels > 1:
            pe[1::2, :] = torch.cos(pos * div[: (channels // 2 + channels % 2)]).transpose(0, 1)[: pe[1::2, :].shape[0]]
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1,C,L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B,C,L), got {tuple(x.shape)}")
        if x.shape[-1] > self.pe.shape[-1]:
            raise ValueError(
                f"latent length {x.shape[-1]} exceeds positional buffer {self.pe.shape[-1]}; "
                "increase max_len in config"
            )
        return x + self.pe[:, :, : x.shape[-1]]


class _CISRJAttentionBlock(nn.Module):
    """Hybrid block: local depthwise conv + gated global attention."""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels, eps=1e-8)
        self.local_dw = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.local_pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.u_proj = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.v_proj = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.h_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.dropout = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,L)
        z = self.norm(x)
        z = z + self.local_pw(self.local_dw(z))

        u = torch.sigmoid(self.u_proj(z))  # (B,2C,L)
        v = self.v_proj(z)  # (B,2C,L)
        h = self.h_proj(z).transpose(1, 2)  # (B,L,C)

        q = self.q_proj(h)  # (B,L,C)
        k = self.k_proj(h)  # (B,L,C)
        scale = (q.shape[-1] ** -0.5) if q.shape[-1] > 0 else 1.0
        attn_logits = torch.matmul(q, k.transpose(1, 2)) * scale  # (B,L,L)
        # Paper uses ReLU^2 style activation for attention scores.
        attn = torch.relu(attn_logits)
        attn = attn * attn
        attn = attn / (torch.sum(attn, dim=-1, keepdim=True) + 1e-8)

        av = torch.matmul(attn, v.transpose(1, 2)).transpose(1, 2)  # (B,2C,L)
        y = self.out_proj(u * av)
        y = self.dropout(y)
        return x + y


class CISRJSN(nn.Module):
    """CISRJ-SN style separator with source masks over IQ mixture."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m = cfg["model_sep"]
        self.n_samples = int(m["n_samples"])
        self.num_sources = int(m.get("num_sources", 4))  # 3 jammer + 1 background
        if self.num_sources < 4:
            raise ValueError(f"num_sources must be >= 4, got {self.num_sources}")

        c = int(m.get("encoder_channels", 512))
        k = int(m.get("encoder_kernel", 16))
        s = int(m.get("encoder_stride", 8))
        blocks = int(m.get("tcn_blocks", 24))
        dropout = float(m.get("tcn_dropout", 0.0))
        self.use_pos = bool(m.get("use_global_pos_enc", True))

        pad = max((k - s) // 2, 0)
        self.encoder = nn.Conv1d(1, c, kernel_size=k, stride=s, padding=pad, bias=False)
        self.enc_norm = nn.GroupNorm(1, c, eps=1e-8)
        self.enc_act = nn.ReLU()
        self.pos_enc = _SinusoidalPositionalEncoding(c, max_len=int(m.get("max_length", 4096)))

        self.blocks = nn.ModuleList([_CISRJAttentionBlock(c, dropout=dropout) for _ in range(blocks)])
        self.mask_head = nn.Conv1d(c, self.num_sources, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"Expected input shape (B,2,N), got {tuple(x.shape)}")
        bsz, _, n = x.shape

        # Paper uses single-channel waveform; here we derive a mono proxy from IQ.
        x_mono = torch.sqrt(torch.clamp(x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1], min=1e-8)).unsqueeze(1)
        h = self.enc_act(self.enc_norm(self.encoder(x_mono)))  # (B,C,L)
        if self.use_pos:
            h = self.pos_enc(h)
        for blk in self.blocks:
            h = blk(h)

        mask_lat = torch.sigmoid(self.mask_head(h))  # (B,S,L)
        mask_time = F.interpolate(mask_lat, size=n, mode="linear", align_corners=False)  # (B,S,N)
        # Normalize source masks to maintain strict mixture consistency.
        weight = mask_time / (torch.sum(mask_time, dim=1, keepdim=True) + 1e-8)  # (B,S,N)
        weight = torch.nan_to_num(weight, nan=1.0 / self.num_sources, posinf=1.0 / self.num_sources, neginf=0.0)

        sources = weight.unsqueeze(2) * x.unsqueeze(1)  # (B,S,2,N)
        # Enforce exact reconstruction: sum(sources) == x
        residual = x - torch.sum(sources, dim=1)
        sources = sources + residual.unsqueeze(1) / float(self.num_sources)
        sources = torch.nan_to_num(sources, nan=0.0, posinf=1e4, neginf=-1e4)

        j_hat = sources[:, :3, :, :]
        b_hat = sources[:, 3, :, :]
        return {
            "j_hat": j_hat,
            "b_hat": b_hat,
            "masks": weight.unsqueeze(2),  # (B,S,1,N)
        }
