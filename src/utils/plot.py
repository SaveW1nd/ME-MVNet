"""Plot utility helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .io import ensure_dir


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 180) -> None:
    """Save matplotlib figure and close it."""
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
