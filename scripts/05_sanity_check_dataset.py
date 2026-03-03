"""Sanity check for generated dataset splits.

Checks:
    - split sample counts: train/val/test = 4200/900/900
    - NF balance per split: train 1400 each class, val/test 300 each class
    - X shape: (B,2,4000), mask shape: (B,4000)
    - physical relation: Tf_s == NF * Tl_s (abs err <= eps)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.utils.io import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/data.yaml")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="runs/sanity_check/figures")
    p.add_argument("--num-plot", type=int, default=3)
    return p.parse_args()


def _load_split(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    return {
        "X": d["X"],
        "mask": d["mask"],
        "Tl_s": d["Tl_s"],
        "Tf_s": d["Tf_s"],
        "NF": d["NF"],
        "JNR_dB": d["JNR_dB"],
    }


def _assert_basic_shapes(split_name: str, d: dict[str, np.ndarray], n_expected: int) -> None:
    x = d["X"]
    mask = d["mask"]
    if x.ndim != 3 or x.shape[1] != 2 or x.shape[2] != n_expected:
        raise AssertionError(f"{split_name}: invalid X shape {x.shape}")
    if mask.shape != (x.shape[0], n_expected):
        raise AssertionError(f"{split_name}: invalid mask shape {mask.shape}")


def _check_nf_balance(split_name: str, nf: np.ndarray, expected_each: int) -> None:
    counts = {int(v): int((nf == v).sum()) for v in [1, 2, 4]}
    print(f"{split_name} NF counts: {counts}")
    if any(v != expected_each for v in counts.values()):
        raise AssertionError(f"{split_name}: NF not balanced, got {counts}")


def _check_tf_relation(split_name: str, tf_s: np.ndarray, nf: np.ndarray, tl_s: np.ndarray, eps: float) -> None:
    err = np.abs(tf_s - nf.astype(np.float32) * tl_s)
    max_err = float(err.max())
    print(f"{split_name} max |Tf-NF*Tl|: {max_err:.3e}")
    if max_err > eps:
        raise AssertionError(f"{split_name}: Tf relation failed, max_err={max_err}")


def _plot_examples(train: dict[str, np.ndarray], out_dir: Path, num_plot: int) -> None:
    x = train["X"]
    mask = train["mask"]
    n = min(num_plot, x.shape[0])
    for i in range(n):
        iq = x[i]
        amp = np.sqrt(iq[0] ** 2 + iq[1] ** 2)
        mk = mask[i]
        t = np.arange(iq.shape[1], dtype=np.int32)

        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t, iq[0], lw=0.8, label="I")
        axes[0].plot(t, iq[1], lw=0.8, label="Q")
        axes[0].legend(loc="upper right")
        axes[0].set_ylabel("IQ")

        axes[1].plot(t, amp, lw=0.8, color="tab:orange")
        axes[1].set_ylabel("Amplitude")

        axes[2].plot(t, mk, lw=0.8, color="tab:green")
        axes[2].set_ylabel("Mask")
        axes[2].set_xlabel("Sample index")
        fig.suptitle(f"Sanity Sample #{i}")
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{i:02d}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_dir = Path(args.data_dir or cfg["dataset"]["output_dir"])
    out_dir = ensure_dir(args.out_dir)
    eps = float(cfg["sanity"]["tf_tl_eps"])

    train = _load_split(data_dir / "train.npz")
    val = _load_split(data_dir / "val.npz")
    test = _load_split(data_dir / "test.npz")

    expected_n = int(cfg["signal"]["n"])
    _assert_basic_shapes("train", train, expected_n)
    _assert_basic_shapes("val", val, expected_n)
    _assert_basic_shapes("test", test, expected_n)

    if train["X"].shape[0] != 4200 or val["X"].shape[0] != 900 or test["X"].shape[0] != 900:
        raise AssertionError(
            "Split sample count mismatch: "
            f"train={train['X'].shape[0]}, val={val['X'].shape[0]}, test={test['X'].shape[0]}"
        )

    _check_nf_balance("train", train["NF"], expected_each=1400)
    _check_nf_balance("val", val["NF"], expected_each=300)
    _check_nf_balance("test", test["NF"], expected_each=300)

    _check_tf_relation("train", train["Tf_s"], train["NF"], train["Tl_s"], eps)
    _check_tf_relation("val", val["Tf_s"], val["NF"], val["Tl_s"], eps)
    _check_tf_relation("test", test["Tf_s"], test["NF"], test["Tl_s"], eps)

    print("JNR levels (train):", sorted(np.unique(train["JNR_dB"]).tolist()))
    print(
        "Tl range (s):",
        f"{float(train['Tl_s'].min()):.6e}",
        "to",
        f"{float(train['Tl_s'].max()):.6e}",
    )
    _plot_examples(train, out_dir=out_dir, num_plot=args.num_plot)
    print(f"Sanity plots saved to: {out_dir}")
    print("Dataset sanity check passed.")


if __name__ == "__main__":
    main()
