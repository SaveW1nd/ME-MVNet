"""Generate composite ISRJ dataset for ME-MVSepPE.

Usage:
    python scripts/10_generate_dataset_composite.py --config configs/data_composite.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.isrj_composite_generator import generate_and_save_composite_dataset
from src.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/data_composite.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    summary = generate_and_save_composite_dataset(cfg)
    print("Composite dataset generation done.")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
