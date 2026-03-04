"""Separation frontend (SepNet) for composite ISRJ."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks_1d import TCNResidualBlock


class SepNet(nn.Module):
    """Conv-TCN-ConvTranspose separator with mixture consistency."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m = cfg["model_sep"]
        self.n_samples = int(m["n_samples"])
        c = int(m["encoder_channels"])
        k = int(m["encoder_kernel"])
        s = int(m["encoder_stride"])
        blocks = int(m["tcn_blocks"])
        d_cycle = int(m["tcn_dilation_cycle"])
        dropout = float(m["tcn_dropout"])

        pad = max((k - s) // 2, 0)
        self.encoder = nn.Conv1d(2, c, kernel_size=k, stride=s, padding=pad, bias=False)
        self.enc_norm = nn.BatchNorm1d(c)
        self.enc_act = nn.GELU()

        tcn_layers = []
        for i in range(blocks):
            dilation = 2 ** (i % d_cycle)
            tcn_layers.append(TCNResidualBlock(c, dilation=dilation, dropout=dropout))
        self.separator = nn.Sequential(*tcn_layers)
        self.mask_proj = nn.Conv1d(c, 4 * c, kernel_size=1)

        self.decoder = nn.ConvTranspose1d(4 * c, 4 * 2, kernel_size=k, stride=s, padding=pad)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"Expected input shape (B,2,N), got {tuple(x.shape)}")
        bsz, _, n = x.shape

        h = self.enc_act(self.enc_norm(self.encoder(x)))  # (B,C,L)
        h = self.separator(h)

        c, l = h.shape[1], h.shape[2]
        mask_logits = self.mask_proj(h).view(bsz, 4, c, l)
        masks = torch.sigmoid(mask_logits)

        h_src = masks * h.unsqueeze(1)  # (B,4,C,L)
        h_src = h_src.reshape(bsz, 4 * c, l)

        decoded = self.decoder(h_src)  # (B,8,N')
        if decoded.shape[-1] > n:
            decoded = decoded[..., :n]
        elif decoded.shape[-1] < n:
            pad_len = n - decoded.shape[-1]
            decoded = nn.functional.pad(decoded, (0, pad_len))

        sources = decoded.view(bsz, 4, 2, n)  # (B,4,2,N)
        j_hat = sources[:, :3, :, :]
        b_hat = sources[:, 3, :, :]

        # Mixture consistency: sum(j_hat)+b_hat == x
        mix_rec = torch.sum(j_hat, dim=1) + b_hat
        residual = x - mix_rec
        j_hat = j_hat + residual.unsqueeze(1) / 4.0
        b_hat = b_hat + residual / 4.0

        return {
            "j_hat": j_hat,
            "b_hat": b_hat,
            "masks": masks,
        }
