"""Analyze Stage-1 separation behavior beyond a single SI-SDR number.

Usage:
    python scripts/16_analyze_sep_behavior.py \
      --ckpt runs/exp_sep_formal_sf5_bg001_finetune_v1/checkpoints/best.pt \
      --model-config configs/model_sep_sf2_wide.yaml \
      --split val \
      --run-dir runs/exp_sep_formal_sf5_bg001_finetune_v1
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_npz_composite import CompositeISRJDataset
from src.models.builders import build_separator
from src.models.pit_perm import align_true_by_perm, best_perm_from_pairwise, pairwise_sep_cost
from src.models.sisdr import si_sdr
from src.utils.io import ensure_dir, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data-config", type=str, default="configs/data_composite.yaml")
    p.add_argument("--model-config", type=str, default="configs/model_sep.yaml")
    p.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--num-cases", type=int, default=6)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--normalize-targets", action="store_true")
    return p.parse_args()


def _select_device(device_arg: str) -> torch.device:
    dev = str(device_arg).lower()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _amp(iq: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(iq[:, 0] * iq[:, 0] + iq[:, 1] * iq[:, 1], min=1e-8))


def _corr_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batch-wise Pearson-like correlation for shape (B,N)."""
    a0 = a - a.mean(dim=1, keepdim=True)
    b0 = b - b.mean(dim=1, keepdim=True)
    num = torch.sum(a0 * b0, dim=1)
    den = torch.sqrt(torch.sum(a0 * a0, dim=1) * torch.sum(b0 * b0, dim=1) + 1e-8)
    return num / den


def _plot_case(
    out_path: Path,
    x: np.ndarray,
    j_true: np.ndarray,
    j_hat: np.ndarray,
    b_hat: np.ndarray,
    nf_true: np.ndarray,
    score: float,
    title_prefix: str,
) -> None:
    t = np.arange(x.shape[-1], dtype=np.int32)
    mix_amp = np.sqrt(x[0] * x[0] + x[1] * x[1])
    bg_amp = np.sqrt(b_hat[0] * b_hat[0] + b_hat[1] * b_hat[1])

    fig, axes = plt.subplots(8, 1, figsize=(11, 11), sharex=True)
    axes[0].plot(t, mix_amp, lw=0.9, color="black")
    axes[0].set_ylabel("|x|")
    axes[0].set_title(f"{title_prefix} | sample SI_SDR_jam={score:.3f} dB | NF={nf_true.tolist()}")
    axes[0].grid(alpha=0.2)

    for k in range(3):
        gt_amp = np.sqrt(j_true[k, 0] * j_true[k, 0] + j_true[k, 1] * j_true[k, 1])
        pd_amp = np.sqrt(j_hat[k, 0] * j_hat[k, 0] + j_hat[k, 1] * j_hat[k, 1])
        axes[1 + 2 * k].plot(t, gt_amp, lw=0.8, color="tab:blue")
        axes[1 + 2 * k].set_ylabel(f"GT j{k+1}")
        axes[1 + 2 * k].grid(alpha=0.2)
        axes[2 + 2 * k].plot(t, pd_amp, lw=0.8, color="tab:orange")
        axes[2 + 2 * k].set_ylabel(f"Pred j{k+1}")
        axes[2 + 2 * k].grid(alpha=0.2)

    axes[7].plot(t, bg_amp, lw=0.8, color="tab:green")
    axes[7].set_ylabel("Pred bg")
    axes[7].set_xlabel("Sample index")
    axes[7].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _collect_records(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float]], np.ndarray, np.ndarray]:
    model.eval()

    # Global aggregates
    diag_vals: list[torch.Tensor] = []
    offdiag_vals: list[torch.Tensor] = []
    collapse_vals: list[torch.Tensor] = []
    silence_vals: list[torch.Tensor] = []
    bg_leak_vals: list[torch.Tensor] = []
    mask_entropy_vals: list[torch.Tensor] = []
    resid_vals: list[torch.Tensor] = []

    pair_sum = torch.zeros((3, 3), dtype=torch.float64)
    pair_cnt = torch.zeros((3, 3), dtype=torch.float64)

    records: list[dict[str, float]] = []
    all_case_scores: list[float] = []
    all_active_counts: list[float] = []

    idx_base = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            bsz = batch["X"].shape[0]

            out = model(batch["X"])
            pair_cost = pairwise_sep_cost(out["j_hat"], batch["J"], batch["NF"])
            perm, _ = best_perm_from_pairwise(pair_cost)
            j_true = align_true_by_perm(batch["J"], perm)
            nf_true = align_true_by_perm(batch["NF"], perm)

            j_hat = out["j_hat"]
            b_hat = out["b_hat"]
            x = batch["X"]

            active = nf_true > 0  # (B,3)
            active_count = active.sum(dim=1).to(torch.float32)

            # Pairwise SI-SDR matrix over active truth sources.
            sdr_mat = torch.zeros((bsz, 3, 3), device=device, dtype=torch.float32)
            for p in range(3):
                for t in range(3):
                    sdr_pt = si_sdr(j_hat[:, p], j_true[:, t]).to(torch.float32)
                    sdr_mat[:, p, t] = sdr_pt
                    active_t = active[:, t]
                    if torch.any(active_t):
                        vals = sdr_pt[active_t].detach().cpu().to(torch.float64)
                        pair_sum[p, t] += vals.sum()
                        pair_cnt[p, t] += float(vals.numel())

            diag = torch.stack([sdr_mat[:, 0, 0], sdr_mat[:, 1, 1], sdr_mat[:, 2, 2]], dim=1)
            active_diag = torch.where(active, diag, torch.zeros_like(diag))
            case_score = active_diag.sum(dim=1) / active_count.clamp_min(1.0)

            for p in range(3):
                diag_vals.append(diag[:, p][active[:, p]].detach().cpu())
            for t in range(3):
                act_t = active[:, t]
                if not torch.any(act_t):
                    continue
                for p in range(3):
                    if p == t:
                        continue
                    offdiag_vals.append(sdr_mat[:, p, t][act_t].detach().cpu())

            env = torch.sqrt(torch.clamp(j_hat[:, :, 0] * j_hat[:, :, 0] + j_hat[:, :, 1] * j_hat[:, :, 1], min=1e-8))
            env_norm = env / torch.sqrt(torch.sum(env * env, dim=2, keepdim=True) + 1e-8)
            sim = torch.einsum("bkn,bjn->bkj", env_norm, env_norm)
            collapse = (sim[:, 0, 1] + sim[:, 0, 2] + sim[:, 1, 2]) / 3.0
            collapse_vals.append(collapse.detach().cpu())

            pred_energy = torch.mean(j_hat * j_hat, dim=(2, 3))
            mix_energy = torch.mean(x * x, dim=(1, 2)) + 1e-8
            inactive = (~active).to(torch.float32)
            inactive_cnt = inactive.sum(dim=1)
            silence_ratio = torch.where(
                inactive_cnt > 0.0,
                (pred_energy * inactive).sum(dim=1) / (inactive_cnt * mix_energy),
                torch.zeros_like(inactive_cnt),
            )
            silence_vals.append(silence_ratio.detach().cpu())

            bg_corr = _corr_1d(_amp(b_hat), _amp(torch.sum(j_true, dim=1)))
            bg_leak_vals.append(bg_corr.detach().cpu())

            masks = out["masks"]
            s = masks.shape[1]
            ent = -torch.sum(masks * torch.log(masks + 1e-8), dim=1) / max(math.log(float(s)), 1e-8)
            mask_entropy = ent.mean(dim=(1, 2))
            mask_entropy_vals.append(mask_entropy.detach().cpu())

            rec = torch.sum(j_hat, dim=1) + b_hat
            resid_ratio = torch.sqrt(torch.mean((rec - x) ** 2, dim=(1, 2))) / torch.sqrt(torch.mean(x * x, dim=(1, 2)) + 1e-8)
            resid_vals.append(resid_ratio.detach().cpu())

            for i in range(bsz):
                rec_i = {
                    "index": float(idx_base + i),
                    "active_count": float(active_count[i].item()),
                    "jam_sisdr_case": float(case_score[i].item()),
                    "collapse_cos": float(collapse[i].item()),
                    "silence_ratio": float(silence_ratio[i].item()),
                    "bg_leak_corr": float(bg_corr[i].item()),
                    "mask_entropy": float(mask_entropy[i].item()),
                    "residual_ratio": float(resid_ratio[i].item()),
                }
                records.append(rec_i)
                all_case_scores.append(rec_i["jam_sisdr_case"])
                all_active_counts.append(rec_i["active_count"])

            idx_base += bsz

    def _cat_mean(vals: list[torch.Tensor]) -> float:
        if not vals:
            return 0.0
        return float(torch.cat(vals, dim=0).mean().item())

    pair_mean = (pair_sum / pair_cnt.clamp_min(1.0)).numpy()
    diag_mean = float(np.mean([pair_mean[0, 0], pair_mean[1, 1], pair_mean[2, 2]]))
    offdiag_mean = float(np.mean([pair_mean[p, t] for p in range(3) for t in range(3) if p != t]))

    summary = {
        "jam_sisdr_diag_mean_db": _cat_mean(diag_vals),
        "jam_sisdr_offdiag_mean_db": _cat_mean(offdiag_vals),
        "jam_sisdr_gap_db": diag_mean - offdiag_mean,
        "collapse_cos_mean": _cat_mean(collapse_vals),
        "silence_ratio_mean": _cat_mean(silence_vals),
        "bg_leak_corr_mean": _cat_mean(bg_leak_vals),
        "mask_entropy_mean": _cat_mean(mask_entropy_vals),
        "residual_ratio_mean": _cat_mean(resid_vals),
        "num_samples": float(len(records)),
        "mean_active_count": float(np.mean(all_active_counts) if all_active_counts else 0.0),
    }
    return summary, records, pair_mean, pair_cnt.numpy()


def _export_case_csv(records: list[dict[str, float]], out_csv: Path) -> None:
    ensure_dir(out_csv.parent)
    fieldnames = [
        "index",
        "active_count",
        "jam_sisdr_case",
        "collapse_cos",
        "silence_ratio",
        "bg_leak_corr",
        "mask_entropy",
        "residual_ratio",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def _plot_selected_cases(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    selected: set[int],
    score_map: dict[int, float],
    out_dir: Path,
    prefix: str,
) -> None:
    model.eval()
    idx_base = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            bsz = batch["X"].shape[0]

            out = model(batch["X"])
            pair_cost = pairwise_sep_cost(out["j_hat"], batch["J"], batch["NF"])
            perm, _ = best_perm_from_pairwise(pair_cost)
            j_true = align_true_by_perm(batch["J"], perm)
            nf_true = align_true_by_perm(batch["NF"], perm)

            for i in range(bsz):
                gidx = idx_base + i
                if gidx not in selected:
                    continue
                _plot_case(
                    out_path=out_dir / f"{prefix}_case_{gidx:05d}.png",
                    x=batch["X"][i].detach().cpu().numpy(),
                    j_true=j_true[i].detach().cpu().numpy(),
                    j_hat=out["j_hat"][i].detach().cpu().numpy(),
                    b_hat=out["b_hat"][i].detach().cpu().numpy(),
                    nf_true=nf_true[i].detach().cpu().numpy(),
                    score=float(score_map[gidx]),
                    title_prefix=prefix,
                )
            idx_base += bsz


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    device = _select_device(args.device)

    ckpt_path = Path(args.ckpt)
    run_dir = Path(args.run_dir) if args.run_dir else ckpt_path.parent.parent
    fig_dir = ensure_dir(run_dir / "figures" / "sep_diagnostics")
    tab_dir = ensure_dir(run_dir / "tables")

    data_dir = Path(data_cfg["dataset"]["output_dir"])
    split_path = data_dir / f"{args.split}.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    ds = CompositeISRJDataset(split_path, normalize_x=True, normalize_targets=bool(args.normalize_targets))
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = build_separator(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)

    summary, records, pair_mean, pair_cnt = _collect_records(model=model, loader=loader, device=device)
    _export_case_csv(records, tab_dir / "sep_case_scores.csv")

    rec_sorted = sorted(records, key=lambda r: r["jam_sisdr_case"])
    num_cases = max(int(args.num_cases), 1)
    worst = rec_sorted[:num_cases]
    best = rec_sorted[-num_cases:]
    worst_idx = {int(r["index"]) for r in worst}
    best_idx = {int(r["index"]) for r in best}
    score_map = {int(r["index"]): float(r["jam_sisdr_case"]) for r in records}

    _plot_selected_cases(
        model=model,
        loader=loader,
        device=device,
        selected=worst_idx,
        score_map=score_map,
        out_dir=fig_dir,
        prefix="worst",
    )
    _plot_selected_cases(
        model=model,
        loader=loader,
        device=device,
        selected=best_idx,
        score_map=score_map,
        out_dir=fig_dir,
        prefix="best",
    )

    save_json(
        {
            "split": args.split,
            "normalize_targets": bool(args.normalize_targets),
            "summary": summary,
            "pairwise_sisdr_mean_db": pair_mean.tolist(),
            "pairwise_sisdr_count": pair_cnt.tolist(),
            "worst_indices": sorted(list(worst_idx)),
            "best_indices": sorted(list(best_idx)),
            "num_cases": num_cases,
            "notes": [
                "collapse_cos: higher means slot outputs are more similar (higher collapse risk)",
                "silence_ratio: higher means more energy leakage into inactive slots",
                "bg_leak_corr: higher means stronger background contamination by jammer envelope",
                "mask_entropy: higher means less confident mask assignment",
            ],
        },
        tab_dir / "sep_diagnostics.json",
    )

    print("Separation behavior diagnostics done.")
    print(f"Summary: {summary}")
    print(f"Tables: {tab_dir / 'sep_diagnostics.json'}")
    print(f"Case scores: {tab_dir / 'sep_case_scores.csv'}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()

