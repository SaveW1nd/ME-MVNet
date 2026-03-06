"""Separation frontend (SepNet) for composite ISRJ."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks_1d import (
    PeriodicGroupingBlock,
    PeriodicContextAggregator,
    SepGateBoundaryRefiner,
    SlotPeriodicContextBank,
    SlotQueryMaskDecoder,
    TCNResidualBlock,
)


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
        self.periodic_context_enabled = bool(m.get("periodic_context_enabled", False))
        self.periodic_grouping_enabled = bool(m.get("periodic_grouping_enabled", False))
        self.slot_periodic_context_enabled = bool(m.get("slot_periodic_context_enabled", False))
        self.slot_periodic_context_scale = float(m.get("slot_periodic_context_scale", 1.0))
        self.sep_gate_enabled = bool(m.get("sep_gate_enabled", False))
        self.sep_gate_bias_scale = float(m.get("sep_gate_bias_scale", 0.35))
        self.sep_gate_refine_enabled = bool(m.get("sep_gate_refine_enabled", False))
        self.sep_edge_enabled = bool(m.get("sep_edge_enabled", False))
        self.slot_query_decoder_enabled = bool(m.get("slot_query_decoder_enabled", False))
        self.mask_mode = str(m.get("mask_mode", "softmax")).lower()
        if self.mask_mode not in {"softmax", "sigmoid_independent"}:
            raise ValueError(f"Unsupported mask_mode: {self.mask_mode}")
        self.n_samples = int(m["n_samples"])
        c = int(m["encoder_channels"])
        k = int(m["encoder_kernel"])
        s = int(m["encoder_stride"])
        self.encoder_stride = s
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
        periodic_lags = [int(v) for v in m.get("periodic_context_lags", [48, 64, 96, 128, 160, 192, 256, 320])]
        self.periodic_lags = periodic_lags
        if self.periodic_context_enabled:
            self.periodic_context = PeriodicContextAggregator(c, lags=periodic_lags)
        else:
            self.periodic_context = None
        if self.periodic_grouping_enabled:
            self.periodic_grouping = PeriodicGroupingBlock(
                c,
                lags=periodic_lags,
                num_groups=int(m.get("periodic_grouping_num_groups", 3)),
                scale_limit=float(m.get("periodic_grouping_scale_limit", 0.25)),
            )
        else:
            self.periodic_grouping = None

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
        if self.slot_query_decoder_enabled:
            self.slot_query_decoder = SlotQueryMaskDecoder(
                channels=c,
                num_slots=self.mask_slots,
                num_heads=int(m.get("slot_query_num_heads", 4)),
                ff_mult=int(m.get("slot_query_ff_mult", 2)),
                dropout=float(m.get("slot_query_dropout", dropout)),
                scale_limit=float(m.get("slot_query_scale_limit", 0.25)),
            )
        else:
            self.slot_query_decoder = None

        if self.slot_periodic_context_enabled:
            self.slot_periodic_context = SlotPeriodicContextBank(c, num_slots=self.mask_slots, lags=periodic_lags)
            self.slot_context_to_logits = nn.Conv1d(
                self.mask_slots * c,
                self.mask_slots * c,
                kernel_size=1,
                groups=self.mask_slots,
                bias=False,
            )
        else:
            self.slot_periodic_context = None
            self.slot_context_to_logits = None

        if self.sep_gate_enabled:
            gate_hidden = int(m.get("sep_gate_hidden", max(32, c // 2)))
            self.sep_gate_head = nn.Sequential(
                nn.Conv1d(c, gate_hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(gate_hidden, 3, kernel_size=1),
            )
            if self.sep_gate_refine_enabled:
                refine_hidden = int(m.get("sep_gate_refine_hidden", gate_hidden))
                self.sep_gate_refiner = SepGateBoundaryRefiner(c, hidden=refine_hidden)
            else:
                self.sep_gate_refiner = None
            if self.sep_edge_enabled and not self.sep_gate_refine_enabled:
                edge_hidden = int(m.get("sep_edge_hidden", gate_hidden))
                self.sep_edge_head = nn.Sequential(
                    nn.Conv1d(c + 3, edge_hidden, kernel_size=5, padding=2, bias=False),
                    nn.BatchNorm1d(edge_hidden),
                    nn.GELU(),
                    nn.Conv1d(edge_hidden, 3, kernel_size=3, padding=1),
                )
            else:
                self.sep_edge_head = None
        else:
            self.sep_gate_head = None
            self.sep_gate_refiner = None
            self.sep_edge_head = None

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
        periodic_weights = None
        if self.periodic_context is not None:
            h, periodic_weights = self.periodic_context(h)
        grouping_weights = None
        if self.periodic_grouping is not None:
            h, grouping_weights = self.periodic_grouping(h)

        c, l = h.shape[1], h.shape[2]
        slot_query_weights = None
        if self.slot_query_decoder is not None:
            mask_logits, slot_query_weights = self.slot_query_decoder(h)
        else:
            mask_logits = self.mask_proj(h).view(bsz, self.mask_slots, c, l)
        sep_gate_logit = None
        sep_edge_logit = None
        if self.sep_gate_head is not None:
            sep_gate_logit_enc = self.sep_gate_head(h)  # (B,3,L)
            if self.sep_gate_refiner is not None:
                sep_gate_logit_enc, sep_edge_logit_enc = self.sep_gate_refiner(h, sep_gate_logit_enc)
            else:
                if self.sep_edge_head is not None:
                    sep_edge_logit_enc = self.sep_edge_head(torch.cat([h, torch.sigmoid(sep_gate_logit_enc)], dim=1))
                else:
                    sep_edge_logit_enc = None
            mask_logits[:, :3, :, :] = mask_logits[:, :3, :, :] + self.sep_gate_bias_scale * sep_gate_logit_enc.unsqueeze(2)
            sep_gate_logit = F.interpolate(
                sep_gate_logit_enc,
                size=n,
                mode="linear",
                align_corners=False,
            )
            if sep_edge_logit_enc is not None:
                up = F.interpolate(
                    sep_edge_logit_enc,
                    size=n,
                    mode="linear",
                    align_corners=False,
                )
                scale = max(math.sqrt(float(n) / max(float(l), 1.0)), 1.0)
                sep_edge_logit = up / scale
        slot_periodic_weights = None
        slot_context = None
        if self.slot_periodic_context is not None:
            slot_context, slot_periodic_weights = self.slot_periodic_context(h)
            slot_bias = self.slot_context_to_logits(slot_context.reshape(bsz, self.mask_slots * c, l))
            slot_bias = slot_bias.view(bsz, self.mask_slots, c, l)
            mask_logits = mask_logits + slot_bias
        if self.mask_mode == "softmax":
            # Source-competitive masks: each time-frequency location is assigned
            # across output slots (default: 3 jammer + 1 background).
            masks = torch.softmax(mask_logits, dim=1)
        else:
            # Independent slot activations reduce forced slot competition
            # when multiple jammers overlap in time-frequency regions.
            masks = torch.sigmoid(mask_logits)

        h_base = h.unsqueeze(1)
        if slot_context is not None:
            h_base = h_base + self.slot_periodic_context_scale * slot_context
        h_src = masks * h_base  # (B,S,C,L)
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
                "periodic_weights": periodic_weights,
                "grouping_weights": grouping_weights,
                "slot_periodic_weights": slot_periodic_weights,
                "slot_query_weights": slot_query_weights,
                "sep_gate_logit": sep_gate_logit,
                "sep_edge_logit": sep_edge_logit,
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
                "periodic_weights": periodic_weights,
                "grouping_weights": grouping_weights,
                "slot_periodic_weights": slot_periodic_weights,
                "slot_query_weights": slot_query_weights,
                "sep_gate_logit": sep_gate_logit,
                "sep_edge_logit": sep_edge_logit,
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
            "periodic_weights": periodic_weights,
            "grouping_weights": grouping_weights,
            "slot_periodic_weights": slot_periodic_weights,
            "slot_query_weights": slot_query_weights,
            "sep_gate_logit": sep_gate_logit,
            "sep_edge_logit": sep_edge_logit,
        }
