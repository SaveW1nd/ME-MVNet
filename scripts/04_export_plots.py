"""Export figures and tables for paper from prediction NPZ.

Usage:
    python scripts/04_export_plots.py --run-dir runs/exp_001 --split test
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.metrics import (
    compute_jnr_bucket_metrics,
    compute_nf_confusion,
    compute_overall_metrics,
)
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--pred", type=str, default=None)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--paper-dir", type=str, default="paper")
    p.add_argument("--mask-num", type=int, default=3)
    return p.parse_args()


def _plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=10, alpha=0.5)
    vmin = float(min(y_true.min(), y_pred.min()))
    vmax = float(max(y_true.max(), y_pred.max()))
    ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=1.2)
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_cm(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1, 2], labels=["1", "2", "4"])
    ax.set_yticks([0, 1, 2], labels=["1", "2", "4"])
    ax.set_xlabel("Predicted NF")
    ax.set_ylabel("True NF")
    ax.set_title("NF Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_jnr_sweep(rows: list[dict], path: Path) -> None:
    j = np.array([r["JNR_dB"] for r in rows], dtype=np.int32)
    tl_mae = np.array([r["Tl_MAE"] for r in rows], dtype=np.float32)
    tf_mae = np.array([r["Tf_MAE"] for r in rows], dtype=np.float32)
    nf_acc = np.array([r["NF_Acc"] for r in rows], dtype=np.float32)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(j, tl_mae, marker="o")
    axes[0].set_ylabel("Tl MAE")
    axes[0].grid(alpha=0.3)

    axes[1].plot(j, tf_mae, marker="o", color="tab:orange")
    axes[1].set_ylabel("Tf MAE")
    axes[1].grid(alpha=0.3)

    axes[2].plot(j, nf_acc, marker="o", color="tab:green")
    axes[2].set_ylabel("NF Acc")
    axes[2].set_xlabel("JNR (dB)")
    axes[2].grid(alpha=0.3)

    fig.suptitle("JNR Sweep Metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_mask_cases(mask_gt: np.ndarray, mask_hat: np.ndarray, path: Path, num: int) -> None:
    n = min(num, mask_gt.shape[0])
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].plot(mask_gt[i], label="mask_gt", lw=1.0)
        axes[i].plot(mask_hat[i], label="mask_hat", lw=1.0, alpha=0.8)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].set_ylabel(f"Sample {i}")
        axes[i].legend(loc="upper right")
        axes[i].grid(alpha=0.2)
    axes[-1].set_xlabel("Sample index")
    fig.suptitle("Mask Prediction Cases")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    pred_path = Path(args.pred) if args.pred else (run_dir / "predictions" / f"{args.split}_pred.npz")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    fig_dir = ensure_dir(run_dir / "figures")
    table_dir = ensure_dir(run_dir / "tables")
    paper_fig_dir = ensure_dir(Path(args.paper_dir) / "figures")
    paper_table_dir = ensure_dir(Path(args.paper_dir) / "tables")

    d = np.load(pred_path, allow_pickle=False)
    tl_true = d["Tl_true"]
    tl_pred = d["Tl_pred"]
    tf_true = d["Tf_true"]
    tf_pred = d["Tf_pred"]
    nf_true = d["NF_true"]
    nf_pred = d["NF_pred"]
    jnr_db = d["JNR_dB"]
    mask_gt = d["mask_gt"]
    mask_hat = d["mask_hat"]

    overall = compute_overall_metrics(
        tl_true=tl_true,
        tl_pred=tl_pred,
        tf_true=tf_true,
        tf_pred=tf_pred,
        nf_true=nf_true,
        nf_pred=nf_pred,
    )
    rows = compute_jnr_bucket_metrics(
        jnr_db=jnr_db,
        tl_true=tl_true,
        tl_pred=tl_pred,
        tf_true=tf_true,
        tf_pred=tf_pred,
        nf_true=nf_true,
        nf_pred=nf_pred,
    )
    cm = compute_nf_confusion(nf_true, nf_pred)

    scatter_tl = fig_dir / "scatter_Tl.png"
    scatter_tf = fig_dir / "scatter_Tf.png"
    cm_path = fig_dir / "cm_NF.png"
    jnr_path = fig_dir / "jnr_sweep.png"
    mask_path = fig_dir / "mask_cases.png"

    _plot_scatter(tl_true, tl_pred, "Tl: Ground Truth vs Prediction", scatter_tl)
    _plot_scatter(tf_true, tf_pred, "Tf: Ground Truth vs Prediction", scatter_tf)
    _plot_cm(cm, cm_path)
    _plot_jnr_sweep(rows, jnr_path)
    _plot_mask_cases(mask_gt, mask_hat, mask_path, num=args.mask_num)

    overall_csv = table_dir / f"{args.split}_metrics_overall.csv"
    jnr_csv = table_dir / f"{args.split}_metrics_by_jnr.csv"
    pd.DataFrame([overall]).to_csv(overall_csv, index=False)
    pd.DataFrame(rows).to_csv(jnr_csv, index=False)

    for p in [scatter_tl, scatter_tf, cm_path, jnr_path, mask_path]:
        shutil.copy2(p, paper_fig_dir / p.name)
    shutil.copy2(overall_csv, paper_table_dir / overall_csv.name)
    shutil.copy2(jnr_csv, paper_table_dir / jnr_csv.name)

    print("Export done.")
    print(f"Figures: {fig_dir}")
    print(f"Tables: {table_dir}")
    print(f"Paper figures: {paper_fig_dir}")
    print(f"Paper tables: {paper_table_dir}")


if __name__ == "__main__":
    main()
