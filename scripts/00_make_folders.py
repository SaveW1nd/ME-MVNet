"""Create project folder skeleton."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    dirs = [
        "configs",
        "scripts",
        "data/raw",
        "data/processed",
        "src/utils",
        "src/data",
        "src/models",
        "src/train",
        "src/eval",
        "runs",
        "paper/figures",
        "paper/tables",
        "tests",
        "data/raw/cisrj_seppe_v1",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("Folder skeleton is ready.")


if __name__ == "__main__":
    main()
