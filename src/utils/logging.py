"""Logging helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .io import ensure_dir


def build_logger(log_file: str | Path) -> logging.Logger:
    """Create a file+stdout logger."""
    log_file = Path(log_file)
    ensure_dir(log_file.parent)

    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON record to a .jsonl file."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
