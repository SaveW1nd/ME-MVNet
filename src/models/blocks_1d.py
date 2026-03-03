"""1D building blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


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
