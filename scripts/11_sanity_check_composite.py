"""Sanity checks for composite dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/data_composite.yaml")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="runs/sanity_composite/figures")
    p.add_argument("--num-plot", type=int, default=3)
    return p.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    return {
        "X": d["X"],
        "J": d["J"],
        "G": d["G"],
        "Tl_us": d["Tl_us"],
        "Ts_us": d["Ts_us"],
        "NF": d["NF"],
        "JNR_dB": d["JNR_dB"],
        "K_active": d["K_active"],
    }


def _check_split(
    name: str,
    d: dict[str, np.ndarray],
    n_expected: int,
    count_expected: int,
    eps: float,
    dual_ratio: float,
) -> None:
    x = d["X"]
    j = d["J"]
    g = d["G"]
    tl = d["Tl_us"]
    ts = d["Ts_us"]
    nf = d["NF"]
    ka = d["K_active"]

    if x.shape != (count_expected, 2, n_expected):
        raise AssertionError(f"{name}: invalid X shape {x.shape}")
    if j.shape != (count_expected, 3, 2, n_expected):
        raise AssertionError(f"{name}: invalid J shape {j.shape}")
    if g.shape != (count_expected, 3, n_expected):
        raise AssertionError(f"{name}: invalid G shape {g.shape}")
    if tl.shape != (count_expected, 3) or ts.shape != (count_expected, 3) or nf.shape != (count_expected, 3):
        raise AssertionError(f"{name}: invalid label shape")

    n2 = int(np.sum(ka == 2))
    n3 = int(np.sum(ka == 3))
    print(f"{name} K_active counts: {{2: {n2}, 3: {n3}}}")
    expected_n2 = int(round(count_expected * dual_ratio))
    expected_n3 = count_expected - expected_n2
    if n2 != expected_n2 or n3 != expected_n3:
        raise AssertionError(
            f"{name}: K_active ratio mismatch. got (2:{n2},3:{n3}) "
            f"expected (2:{expected_n2},3:{expected_n3})"
        )

    # Dual samples must have third source silence.
    idx_dual = np.where(ka == 2)[0]
    if idx_dual.size > 0:
        if not np.allclose(j[idx_dual, 2], 0.0):
            raise AssertionError(f"{name}: dual sample third J is not zero")
        if not np.all(g[idx_dual, 2] == 0):
            raise AssertionError(f"{name}: dual sample third G is not zero")
        if not np.all(nf[idx_dual, 2] == 0):
            raise AssertionError(f"{name}: dual sample third NF is not zero")
        if not np.allclose(tl[idx_dual, 2], 0.0):
            raise AssertionError(f"{name}: dual sample third Tl is not zero")
        if not np.allclose(ts[idx_dual, 2], 0.0):
            raise AssertionError(f"{name}: dual sample third Ts is not zero")

    active = nf > 0
    ts_ref = (nf.astype(np.float32) + 1.0) * tl
    max_err = float(np.max(np.abs(ts[active] - ts_ref[active]))) if np.any(active) else 0.0
    print(f"{name} max |Ts-(NF+1)Tl| (active): {max_err:.3e}")
    if max_err > eps:
        raise AssertionError(f"{name}: Ts relation failed (max err {max_err})")

    nf_vals = sorted(np.unique(nf).tolist())
    print(f"{name} NF values: {nf_vals}")


def _plot_examples(train: dict[str, np.ndarray], out_dir: Path, num_plot: int) -> None:
    x = train["X"]
    j = train["J"]
    g = train["G"]
    ka = train["K_active"]
    n = min(num_plot, x.shape[0])
    for i in range(n):
        t = np.arange(x.shape[-1], dtype=np.int32)
        mix_amp = np.sqrt(x[i, 0] ** 2 + x[i, 1] ** 2)

        fig, axes = plt.subplots(5, 1, figsize=(11, 9), sharex=True)
        axes[0].plot(t, x[i, 0], lw=0.7, label="mix I")
        axes[0].plot(t, x[i, 1], lw=0.7, label="mix Q")
        axes[0].legend(loc="upper right")
        axes[0].set_title(f"Sample {i} | K_active={int(ka[i])}")

        axes[1].plot(t, mix_amp, lw=0.8, color="black")
        axes[1].set_ylabel("|x|")

        for k in range(3):
            jam_amp = np.sqrt(j[i, k, 0] ** 2 + j[i, k, 1] ** 2)
            ax = axes[2 + k]
            ax.plot(t, jam_amp, lw=0.7, label=f"|j{k+1}|")
            ax.plot(t, g[i, k], lw=0.7, label=f"g{k+1}", alpha=0.8)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Sample index")
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{i:02d}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_dir = Path(args.data_dir or cfg["dataset"]["output_dir"])
    out_dir = ensure_dir(args.out_dir)

    n_expected = int(cfg["signal"]["n"])
    eps = float(cfg["sanity"]["ts_tl_eps"])
    expected = cfg["split"]
    dual_ratio = float(cfg["grid"]["k_active_dual_ratio"])

    train = _load_npz(data_dir / "train.npz")
    val = _load_npz(data_dir / "val.npz")
    test = _load_npz(data_dir / "test.npz")

    _check_split("train", train, n_expected, int(expected["train"]), eps, dual_ratio)
    _check_split("val", val, n_expected, int(expected["val"]), eps, dual_ratio)
    _check_split("test", test, n_expected, int(expected["test"]), eps, dual_ratio)

    _plot_examples(train, out_dir=out_dir, num_plot=args.num_plot)
    print(f"Sanity figures saved to: {out_dir}")
    print("Composite dataset sanity check passed.")


if __name__ == "__main__":
    main()
