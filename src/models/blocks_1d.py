"""1D building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct1d(nn.Module):
    """Conv1d + BatchNorm1d + GELU."""

    def __init__(self, c_in: int, c_out: int, k: int, s: int = 1, d: int = 1) -> None:
        super().__init__()
        p = ((k - 1) // 2) * d
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=p, dilation=d, bias=False),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TCNResidualBlock(nn.Module):
    """Residual dilated conv block for temporal modeling."""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = ConvBNAct1d(channels, channels, k=3, d=dilation)
        self.conv2 = ConvBNAct1d(channels, channels, k=3, d=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        return x + out


class PeriodicContextAggregator(nn.Module):
    """Fuse lag-aligned history to emphasize periodic retransmission structure."""

    def __init__(self, channels: int, lags: list[int]) -> None:
        super().__init__()
        clean_lags = sorted({int(v) for v in lags if int(v) > 0})
        if not clean_lags:
            raise ValueError("PeriodicContextAggregator requires at least one positive lag")
        self.lags = clean_lags

        self.context_proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.gate_proj = nn.Conv1d(channels * 2, channels, kernel_size=1)

    @staticmethod
    def _shift_right(x: torch.Tensor, lag: int) -> torch.Tensor:
        if lag <= 0:
            return x
        if lag >= x.shape[-1]:
            return torch.zeros_like(x)
        return F.pad(x[..., :-lag], (lag, 0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, channels, length = x.shape
        shifted = []
        scores = []
        x_norm = F.normalize(x, dim=1, eps=1e-6)

        for lag in self.lags:
            if lag >= length:
                continue
            lagged = self._shift_right(x, lag)
            shifted.append(lagged)

            cur = x_norm[..., lag:]
            ref = F.normalize(lagged[..., lag:], dim=1, eps=1e-6)
            score = torch.mean(cur * ref, dim=(1, 2))
            scores.append(score)

        if not shifted:
            empty = torch.zeros(bsz, 0, device=x.device, dtype=x.dtype)
            return x, empty

        score_t = torch.stack(scores, dim=1)
        weight_t = torch.softmax(score_t, dim=1)

        context = torch.zeros_like(x)
        for idx, lagged in enumerate(shifted):
            context = context + weight_t[:, idx].view(bsz, 1, 1) * lagged

        context = self.context_proj(context)
        gate = torch.sigmoid(self.gate_proj(torch.cat([x, context], dim=1)))
        out = x + gate * self.out_proj(context)
        return out, weight_t


class PeriodicGroupingBlock(nn.Module):
    """Group lag-aligned periodic evidence before separator decoding."""

    def __init__(
        self,
        channels: int,
        lags: list[int],
        num_groups: int = 3,
        scale_limit: float = 0.25,
    ) -> None:
        super().__init__()
        clean_lags = sorted({int(v) for v in lags if int(v) > 0})
        if not clean_lags:
            raise ValueError("PeriodicGroupingBlock requires at least one positive lag")
        self.channels = int(channels)
        self.lags = clean_lags
        self.num_groups = int(num_groups)
        self.scale_limit = float(scale_limit)

        self.group_query = nn.Linear(self.channels, self.num_groups * self.channels)
        self.group_proj = nn.Sequential(
            nn.Conv1d(self.num_groups * self.channels, self.channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.channels),
            nn.GELU(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv1d(self.channels, self.channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.channels),
            nn.GELU(),
        )
        self.gate_proj = nn.Conv1d(self.channels * 2, self.channels, kernel_size=1)

    @staticmethod
    def _shift_right(x: torch.Tensor, lag: int) -> torch.Tensor:
        if lag <= 0:
            return x
        if lag >= x.shape[-1]:
            return torch.zeros_like(x)
        return F.pad(x[..., :-lag], (lag, 0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x (B,C,L), got {tuple(x.shape)}")
        bsz, channels, length = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected channels={self.channels}, got {channels}")

        lag_bank = []
        lag_desc = []
        for lag in self.lags:
            if lag >= length:
                continue
            lagged = self._shift_right(x, lag)
            lag_bank.append(lagged)
            lag_desc.append(lagged.mean(dim=-1))

        if not lag_bank:
            empty = torch.zeros(bsz, self.num_groups, 0, device=x.device, dtype=x.dtype)
            return x, empty

        bank = torch.stack(lag_bank, dim=1)  # (B,M,C,L)
        desc = torch.stack(lag_desc, dim=1)  # (B,M,C)
        query = self.group_query(x.mean(dim=-1)).view(bsz, self.num_groups, channels)
        score = torch.einsum("bgc,bmc->bgm", query, desc) / (float(channels) ** 0.5)
        weight = torch.softmax(score, dim=-1)
        group_context = torch.einsum("bgm,bmcl->bgcl", weight, bank)

        scale = self.scale_limit * torch.tanh(query).unsqueeze(-1)
        group_context = group_context * (1.0 + scale)
        group_context = group_context.reshape(bsz, self.num_groups * channels, length)
        group_context = self.group_proj(group_context)

        gate = torch.sigmoid(self.gate_proj(torch.cat([x, group_context], dim=1)))
        out = x + gate * self.out_proj(group_context)
        return out, weight


class SlotPeriodicContextBank(nn.Module):
    """Build slot-specific lag-aligned contexts from a shared lag bank."""

    def __init__(self, channels: int, num_slots: int, lags: list[int]) -> None:
        super().__init__()
        clean_lags = sorted({int(v) for v in lags if int(v) > 0})
        if not clean_lags:
            raise ValueError("SlotPeriodicContextBank requires at least one positive lag")
        self.lags = clean_lags
        self.num_slots = int(num_slots)
        self.channels = int(channels)

        self.query_proj = nn.Linear(self.channels, self.num_slots * self.channels)
        self.context_proj = nn.Sequential(
            nn.Conv1d(self.channels, self.channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.channels),
            nn.GELU(),
        )

    @staticmethod
    def _shift_right(x: torch.Tensor, lag: int) -> torch.Tensor:
        if lag <= 0:
            return x
        if lag >= x.shape[-1]:
            return torch.zeros_like(x)
        return F.pad(x[..., :-lag], (lag, 0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, channels, length = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected channels={self.channels}, got {channels}")

        lag_bank = []
        lag_desc = []
        for lag in self.lags:
            if lag >= length:
                continue
            lagged = self._shift_right(x, lag)
            lag_bank.append(lagged)
            lag_desc.append(lagged.mean(dim=-1))

        if not lag_bank:
            empty_w = torch.zeros(bsz, self.num_slots, 0, device=x.device, dtype=x.dtype)
            empty_c = torch.zeros(bsz, self.num_slots, channels, length, device=x.device, dtype=x.dtype)
            return empty_c, empty_w

        bank = torch.stack(lag_bank, dim=1)  # (B,L,C,T)
        desc = torch.stack(lag_desc, dim=1)  # (B,L,C)
        query = self.query_proj(x.mean(dim=-1)).view(bsz, self.num_slots, channels)
        score = torch.einsum("bsc,blc->bsl", query, desc) / (float(channels) ** 0.5)
        weight = torch.softmax(score, dim=-1)
        context = torch.einsum("bsl,blct->bsct", weight, bank)

        flat = context.reshape(bsz * self.num_slots, channels, length)
        flat = self.context_proj(flat)
        context = flat.view(bsz, self.num_slots, channels, length)
        return context, weight


class SlotQueryMaskDecoder(nn.Module):
    """Decode slot-specific mask logits from shared temporal features via slot queries."""

    def __init__(
        self,
        channels: int,
        num_slots: int,
        num_heads: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.1,
        scale_limit: float = 0.25,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        self.channels = int(channels)
        self.num_slots = int(num_slots)
        self.scale_limit = float(scale_limit)

        self.slot_queries = nn.Parameter(torch.randn(1, self.num_slots, self.channels) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.query_norm1 = nn.LayerNorm(self.channels)
        self.query_ffn = nn.Sequential(
            nn.Linear(self.channels, self.channels * int(ff_mult)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.channels * int(ff_mult), self.channels),
        )
        self.query_norm2 = nn.LayerNorm(self.channels)
        self.slot_affine = nn.Linear(self.channels, self.channels * 2)
        self.slot_proj = nn.Conv1d(
            self.num_slots * self.channels,
            self.num_slots * self.channels,
            kernel_size=1,
            groups=self.num_slots,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x (B,C,L), got {tuple(x.shape)}")
        bsz, channels, length = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected channels={self.channels}, got {channels}")

        seq = x.transpose(1, 2)  # (B,L,C)
        q0 = self.slot_queries.expand(bsz, -1, -1)
        q_attn, attn_w = self.cross_attn(q0, seq, seq, need_weights=True, average_attn_weights=True)
        q = self.query_norm1(q0 + q_attn)
        q = self.query_norm2(q + self.query_ffn(q))

        affine = self.slot_affine(q).view(bsz, self.num_slots, 2, self.channels)
        scale = self.scale_limit * torch.tanh(affine[:, :, 0, :]).unsqueeze(-1)
        bias = affine[:, :, 1, :].unsqueeze(-1)

        slot_feat = x.unsqueeze(1) * (1.0 + scale) + bias
        mask_logits = self.slot_proj(slot_feat.reshape(bsz, self.num_slots * self.channels, length))
        mask_logits = mask_logits.view(bsz, self.num_slots, self.channels, length)
        mask_logits = mask_logits + attn_w.unsqueeze(2)
        return mask_logits, attn_w


class SepGateBoundaryRefiner(nn.Module):
    """Refine slot occupancy logits and predict boundary transitions."""

    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        in_ch = int(channels) + 3
        hid = int(hidden)
        self.refine = nn.Sequential(
            nn.Conv1d(in_ch, hid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hid),
            nn.GELU(),
            nn.Conv1d(hid, 3, kernel_size=3, padding=1),
        )
        self.edge = nn.Sequential(
            nn.Conv1d(in_ch, hid, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hid),
            nn.GELU(),
            nn.Conv1d(hid, 3, kernel_size=3, padding=1),
        )

    def forward(self, h: torch.Tensor, gate_logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_prob = torch.sigmoid(gate_logit)
        refine_in = torch.cat([h, gate_prob], dim=1)
        gate_logit = gate_logit + self.refine(refine_in)
        edge_in = torch.cat([h, torch.sigmoid(gate_logit)], dim=1)
        edge_logit = self.edge(edge_in)
        return gate_logit, edge_logit
