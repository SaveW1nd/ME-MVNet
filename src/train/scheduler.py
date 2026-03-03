"""Scheduler builder."""

from __future__ import annotations

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Use cosine annealing for stable training."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=max(total_epochs, 1),
        eta_min=1e-6,
    )
