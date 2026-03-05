"""Export ME-MVSepPE figures/tables to run and paper folders."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--pred", type=str, default=None)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--paper-dir", type=str, default="paper")
    p.add_argument("--num-cases", type=int, default=3)
    return p.parse_args()


def _scatter(true_v: np.ndarray, pred_v: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_v, pred_v, s=8, alpha=0.45)
    lo = float(min(true_v.min(), pred_v.min()))
    hi = float(max(true_v.max(), pred_v.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
    ax.set_xlabel("Ground Truth (us)")
    ax.set_ylabel("Prediction (us)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_cm(nf_true: np.ndarray, nf_pred: np.ndarray, out_path: Path) -> None:
    cm = confusion_matrix(nf_true.reshape(-1), nf_pred.reshape(-1), labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1, 2, 3], labels=["0", "1", "2", "3"])
    ax.set_yticks([0, 1, 2, 3], labels=["0", "1", "2", "3"])
    ax.set_xlabel("Predicted NF")
    ax.set_ylabel("True NF")
    ax.set_title("NF Confusion Matrix (4-class)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_a_total_vs_jnr(jnr_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(jnr_csv)
    if df.empty:
        raise ValueError(f"No rows in {jnr_csv}")
    x = 0.5 * (df["JNR_lo"].to_numpy(dtype=np.float32) + df["JNR_hi"].to_numpy(dtype=np.float32))
    y = df["A_total"].to_numpy(dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("JNR (dB)")
    ax.set_ylabel("A_total")
    ax.set_title("A_total vs JNR")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_separation_cases(
    x: np.ndarray,
    j_true: np.ndarray,
    j_hat: np.ndarray,
    out_dir: Path,
    num_cases: int,
) -> list[Path]:
    n = min(num_cases, x.shape[0])
    out_paths = []
    for i in range(n):
        t = np.arange(x.shape[-1], dtype=np.int32)
        mix_amp = np.sqrt(x[i, 0] ** 2 + x[i, 1] ** 2)

        fig, axes = plt.subplots(7, 1, figsize=(11, 10), sharex=True)
        axes[0].plot(t, mix_amp, lw=0.9, color="black")
        axes[0].set_ylabel("|x|")
        axes[0].set_title(f"Separation Case #{i}")

        for k in range(3):
            gt_amp = np.sqrt(j_true[i, k, 0] ** 2 + j_true[i, k, 1] ** 2)
            pd_amp = np.sqrt(j_hat[i, k, 0] ** 2 + j_hat[i, k, 1] ** 2)

            axes[1 + 2 * k].plot(t, gt_amp, lw=0.8, color="tab:blue")
            axes[1 + 2 * k].set_ylabel(f"GT j{k+1}")
            axes[1 + 2 * k].grid(alpha=0.2)

            axes[2 + 2 * k].plot(t, pd_amp, lw=0.8, color="tab:orange")
            axes[2 + 2 * k].set_ylabel(f"Pred j{k+1}")
            axes[2 + 2 * k].grid(alpha=0.2)

        axes[-1].set_xlabel("Sample index")
        fig.tight_layout()
        p = out_dir / f"separation_case_{i:02d}.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        out_paths.append(p)
    return out_paths


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    fig_dir = ensure_dir(run_dir / "figures")
    table_dir = ensure_dir(run_dir / "tables")
    pred_path = Path(args.pred) if args.pred else (run_dir / "predictions" / f"{args.split}_pred.npz")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")

    d = np.load(pred_path, allow_pickle=False)
    tl_true = d["Tl_true_us"]
    tl_pred = d["Tl_pred_us"]
    ts_true = d["Ts_true_us"]
    ts_pred = d["Ts_pred_us"]
    nf_true = d["NF_true"]
    nf_pred = d["NF_pred"]
    x = d["X"]
    j_true = d["j_true"]
    j_hat = d["j_hat"]

    active = nf_true > 0
    scatter_tl = fig_dir / "scatter_Tl_us.png"
    scatter_ts = fig_dir / "scatter_Ts_us.png"
    cm_path = fig_dir / "cm_NF_4class.png"
    a_total_path = fig_dir / "a_total_vs_jnr.png"

    _scatter(tl_true[active], tl_pred[active], "Tl Scatter (active sources)", scatter_tl)
    _scatter(ts_true[active], ts_pred[active], "Ts Scatter (active sources)", scatter_ts)
    _plot_cm(nf_true, nf_pred, cm_path)
    _plot_a_total_vs_jnr(table_dir / "test_metrics_by_jnr.csv", a_total_path)
    case_paths = _plot_separation_cases(x=x, j_true=j_true, j_hat=j_hat, out_dir=fig_dir, num_cases=args.num_cases)

    paper_dir = Path(args.paper_dir)
    paper_fig = ensure_dir(paper_dir / "figures")
    paper_tbl = ensure_dir(paper_dir / "tables")

    for p in [scatter_tl, scatter_ts, cm_path, a_total_path, *case_paths]:
        shutil.copy2(p, paper_fig / p.name)
    for p in [
        table_dir / "test_metrics.json",
        table_dir / "test_metrics_by_jnr.csv",
        table_dir / "test_metrics_by_kactive.csv",
        table_dir / "metrics_by_nf.csv",
        table_dir / "metrics_cond_nf_correct.csv",
    ]:
        if p.exists():
            shutil.copy2(p, paper_tbl / p.name)

    print("SEPPE plot export done.")
    print(f"Figures: {fig_dir}")
    print(f"Tables: {table_dir}")
    print(f"Paper figures: {paper_fig}")
    print(f"Paper tables: {paper_tbl}")


if __name__ == "__main__":
    main()
