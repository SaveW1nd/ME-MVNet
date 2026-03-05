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
        self.jam_only = bool(m.get("jam_only", False))
        self.num_out = 3 if self.jam_only else 4
        self.decoder_grouped = bool(m.get("decoder_grouped", False))
        self.bg_dedicated = bool(m.get("bg_dedicated", False)) and not self.jam_only
        self.bg_residual_ratio = float(m.get("bg_residual_ratio", 0.2))
        self.bg_residual_scale = float(m.get("bg_residual_scale", 1.0))
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
        if self.bg_dedicated:
            # Jammer slots are separated first; background has a dedicated branch.
            self.mask_slots = 3
            self.mask_proj = nn.Conv1d(c, self.mask_slots * c, kernel_size=1)
            dec_groups = self.mask_slots if self.decoder_grouped else 1
            self.decoder = nn.ConvTranspose1d(
                self.mask_slots * c,
                self.mask_slots * 2,
                kernel_size=k,
                stride=s,
                padding=pad,
                groups=dec_groups,
            )
            self.bg_proj = nn.Conv1d(c, c, kernel_size=1, bias=False)
            self.bg_decoder = nn.ConvTranspose1d(c, 2, kernel_size=k, stride=s, padding=pad)
        else:
            self.mask_slots = self.num_out
            self.mask_proj = nn.Conv1d(c, self.mask_slots * c, kernel_size=1)
            dec_groups = self.mask_slots if self.decoder_grouped else 1
            self.decoder = nn.ConvTranspose1d(
                self.mask_slots * c,
                self.mask_slots * 2,
                kernel_size=k,
                stride=s,
                padding=pad,
                groups=dec_groups,
            )
            self.bg_proj = None
            self.bg_decoder = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError(f"Expected input shape (B,2,N), got {tuple(x.shape)}")
        bsz, _, n = x.shape

        def _match_len(t: torch.Tensor, target_len: int) -> torch.Tensor:
            if t.shape[-1] > target_len:
                return t[..., :target_len]
            if t.shape[-1] < target_len:
                return nn.functional.pad(t, (0, target_len - t.shape[-1]))
            return t

        h = self.enc_act(self.enc_norm(self.encoder(x)))  # (B,C,L)
        h = self.separator(h)

        c, l = h.shape[1], h.shape[2]
        mask_logits = self.mask_proj(h).view(bsz, self.mask_slots, c, l)
        # Source-competitive masks: each time-frequency location is assigned
        # across output slots (default: 3 jammer + 1 background).
        masks = torch.softmax(mask_logits, dim=1)

        h_src = masks * h.unsqueeze(1)  # (B,S,C,L)
        h_src = h_src.reshape(bsz, self.mask_slots * c, l)

        decoded = self.decoder(h_src)  # (B,8,N')
        decoded = _match_len(decoded, n)

        if self.bg_dedicated:
            j_hat = decoded.view(bsz, 3, 2, n)
            b_hat = self.bg_decoder(self.bg_proj(h))
            b_hat = _match_len(b_hat, n)

            mix_rec = torch.sum(j_hat, dim=1) + b_hat
            residual = x - mix_rec

            jam_energy = torch.mean(j_hat * j_hat, dim=(2, 3))  # (B,3)
            jam_energy = torch.nan_to_num(jam_energy, nan=0.0, posinf=1e6, neginf=0.0)
            jam_weight = jam_energy / (torch.sum(jam_energy, dim=1, keepdim=True) + 1e-8)
            jam_weight = torch.nan_to_num(jam_weight, nan=1.0 / 3.0, posinf=1.0 / 3.0, neginf=1.0 / 3.0)

            bg_ratio = min(max(self.bg_residual_ratio, 0.0), 1.0)
            j_hat = j_hat + residual.unsqueeze(1) * jam_weight.unsqueeze(-1).unsqueeze(-1) * (1.0 - bg_ratio)
            b_hat = b_hat + residual * bg_ratio

            j_hat = torch.nan_to_num(j_hat, nan=0.0, posinf=1e4, neginf=-1e4)
            b_hat = torch.nan_to_num(b_hat, nan=0.0, posinf=1e4, neginf=-1e4)
            return {
                "j_hat": j_hat,
                "b_hat": b_hat,
                "masks": masks,
            }

        sources = decoded.view(bsz, self.num_out, 2, n)  # (B,S,2,N)

        if self.jam_only:
            j_hat = torch.nan_to_num(sources, nan=0.0, posinf=1e4, neginf=-1e4)
            b_hat = x - torch.sum(j_hat, dim=1)
            b_hat = torch.nan_to_num(b_hat, nan=0.0, posinf=1e4, neginf=-1e4)
            return {
                "j_hat": j_hat,
                "b_hat": b_hat,
                "masks": masks,
            }

        # Mixture consistency: sum(j_hat)+b_hat == x
        mix_rec = torch.sum(sources, dim=1)
        residual = x - mix_rec

        # Residual is distributed by source energy instead of uniform split.
        src_energy = torch.mean(sources * sources, dim=(2, 3))  # (B,4)
        src_energy = torch.nan_to_num(src_energy, nan=0.0, posinf=1e6, neginf=0.0)
        if src_energy.shape[1] == 4 and self.bg_residual_scale != 1.0:
            # Optionally suppress residual assignment to background slot (index=3).
            src_energy = src_energy.clone()
            src_energy[:, 3] = src_energy[:, 3] * max(self.bg_residual_scale, 0.0)
        src_weight = src_energy / (torch.sum(src_energy, dim=1, keepdim=True) + 1e-8)
        src_weight = torch.nan_to_num(src_weight, nan=0.25, posinf=0.25, neginf=0.25)
        sources = sources + residual.unsqueeze(1) * src_weight.unsqueeze(-1).unsqueeze(-1)
        sources = torch.nan_to_num(sources, nan=0.0, posinf=1e4, neginf=-1e4)

        j_hat = sources[:, :3, :, :]
        b_hat = sources[:, 3, :, :]

        return {
            "j_hat": j_hat,
            "b_hat": b_hat,
            "masks": masks,
        }
