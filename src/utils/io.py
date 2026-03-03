"""I/O helpers for configs and experiment folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML to dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Save dictionary as YAML with UTF-8 encoding."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save dictionary as JSON UTF-8."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def next_experiment_dir(root: str | Path, prefix: str = "exp_") -> Path:
    """Create the next run directory as exp_XXX."""
    root = ensure_dir(root)
    existing = []
    for p in root.glob(f"{prefix}*"):
        if p.is_dir():
            suffix = p.name.replace(prefix, "")
            if suffix.isdigit():
                existing.append(int(suffix))
    idx = (max(existing) + 1) if existing else 1
    run_dir = root / f"{prefix}{idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
