"""Optimizer builder."""

from __future__ import annotations

import torch


def build_optimizer(model: torch.nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    """Create AdamW optimizer."""
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
